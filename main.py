# main.py
import json
import os
import asyncio
import uvicorn
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from loguru import logger
import openai
from chromadb.config import Settings
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Import the workout plan logic
from workout_plan_logic import generate_workout_plan_with_ai

# Make sure to import your router:
from routers.workout_plan import router as workout_plan_router

# -----------------------------------------------------------------------------
# API Configuration
# -----------------------------------------------------------------------------

# Set your OpenAI API key
API_KEY = "sk-RSlhENXN4JW0Hq5v060fDaB48c0d4888B402Ea723949300c"  # Replace with your actual API key
BASE_URL = "https://api.gpt.ge/v1/"

# Initialize OpenAI settings
openai.api_key = API_KEY
openai.base_url = BASE_URL
openai.default_headers = {"x-foo": "true"}

# -----------------------------------------------------------------------------
# Initialize FastAPI app and templates
# -----------------------------------------------------------------------------

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates if necessary
# app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Include the router so that /adjust_workout_plan is registered:
app.include_router(workout_plan_router)

# -----------------------------------------------------------------------------
# Setup LangChain components for PDF Chatbot
# -----------------------------------------------------------------------------

# Create directories if they do not exist
if not os.path.exists('files'):
    os.mkdir('files')
if not os.path.exists('vectorstore_db'):
    os.mkdir('vectorstore_db')



# Define prompt template
template = """
Answer the question based on the context, in a concise manner, in markdown and using bullet points where applicable.

Context: {context}
History: {history}

Question: {question}
Answer:
"""

prompt = PromptTemplate(
    input_variables=["history", "context", "question"],
    template=template,
)

memory = ConversationBufferMemory(
    memory_key="history",
    return_messages=True,
    input_key="question"
)

# Initialize embeddings model
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",  # Specifically use embedding model
    api_key=API_KEY,
    base_url=BASE_URL,
    default_headers=openai.default_headers
)

# Initialize vector store with settings
chroma_settings = Settings(
    anonymized_telemetry=False,
    is_persistent=True
)

vector_store = Chroma(
    persist_directory='vectorstore_db',
    embedding_function=embeddings,
    client_settings=chroma_settings
)

# Initialize chat model
llm = ChatOpenAI(
    model="gpt-3.5-turbo",  # Use chat model instead of embedding model
    api_key=API_KEY,
    base_url=BASE_URL,
    default_headers=openai.default_headers,
    streaming=True,
    verbose=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)

# Configure text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=200,
    length_function=len
)

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def delete_document(file_path):
    try:
        coll = vector_store.get()
        ids_to_del = [id for idx, id in enumerate(coll['ids'])
                      if coll['metadatas'][idx]['source_file_path'] == file_path]
        if ids_to_del:
            vector_store._collection.delete(ids_to_del)
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")

def handle_get_documents():
    try:
        coll = vector_store.get()
        source_file_paths = [metadata['source_file_path'] for metadata in coll['metadatas']]
        return list(set(source_file_paths))
    except Exception as e:
        logger.error(f"Error getting documents: {str(e)}")
        return []

def upload_pdf_to_vectorstore_db(file_path: str):
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load_and_split(text_splitter)
        for doc in docs:
            doc.metadata = {"source_file_path": file_path.split("/")[-1]}
        vector_store.add_documents(docs)
        print(f"Successfully uploaded {len(docs)} documents from {file_path}")
    except Exception as e:
        logger.error(f"Error uploading PDF: {str(e)}")
        raise

# -----------------------------------------------------------------------------
# PDF Chatbot Endpoints
# -----------------------------------------------------------------------------

@app.get("/get_documents/")
def get_documents():
    documents = handle_get_documents()
    return {"data": documents}

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    file_location = os.path.join("files", file.filename)
    logger.info(f"Received PDF: {file_location}")

    # Save the uploaded file
    with open(file_location, "wb") as f:
        f.write(await file.read())

    upload_pdf_to_vectorstore_db(file_location)
    return {"message": "PDF uploaded and processed successfully"}

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await websocket.accept()
        retriever = vector_store.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "verbose": True,
                "prompt": prompt,
                "memory": memory,
            }
        )
        
        while True:
            try:
                user_input = await websocket.receive_text()
                response = qa_chain.invoke({"query": user_input})
                answer = response["result"]
                context = response["source_documents"]

                # Stream response word by word
                for chunk in answer.split(" "):
                    await websocket.send_text(json.dumps({
                        "event_type": "answer",
                        "data": chunk + " "
                    }, ensure_ascii=False))
                    await asyncio.sleep(0.05)

                # Send source document information
                file_names = [f"**{doc.metadata.get('source_file_path', '')}**"
                              for doc in context]
                if file_names:
                    document = "\n".join(list(set(file_names)))
                    await websocket.send_text(json.dumps({
                        "event_type": "answer",
                        "data": f"\n\nSource PDF:\n\n{document}"
                    }, ensure_ascii=False))

            except Exception as e:
                logger.error(f"Error in chat loop: {str(e)}")
                await websocket.send_text(json.dumps({
                    "event_type": "error",
                    "data": str(e)
                }))
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        logger.info("WebSocket connection closed")
        try:
            await websocket.close()
        except Exception:
            pass


@app.on_event("shutdown")
async def shutdown_event():
    # No need to call vector_store._client.close()
    logger.info("Application is shutting down")

# -----------------------------------------------------------------------------
# Workout Planner Endpoints
# -----------------------------------------------------------------------------

@app.get("/workout_planner", response_class=HTMLResponse)
async def workout_planner_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate_workout_plan")
async def generate_workout_plan(
    request: Request,
    name: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    height: float = Form(...),
    weight: float = Form(...),
    goal: int = Form(...),
    activity_level: int = Form(...),
    workout_type: int = Form(...),
    experience_level: int = Form(...),
    equipment: int = Form(...),
    time_available: int = Form(...)
):
    try:
        user_data = {
            'name': name,
            'age': age,
            'gender': gender,
            'height': height,
            'weight': weight,
            'goal': goal,
            'activity_level': activity_level,
            'workout_type': workout_type,
            'experience_level': experience_level,
            'equipment': equipment,
            'time_available': time_available
        }

        # Validate all fields quickly
        if not all(str(v).strip() for v in user_data.values()):
            return JSONResponse({"error": "All fields are required."}, status_code=400)

        workout_plan = generate_workout_plan_with_ai(user_data)

        # On success, return JSON with the plan
        return JSONResponse({"plan": workout_plan, "name": user_data["name"]})

    except ValueError as e:
        # Return a 400 for any invalid user input
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        logger.error(f"Error generating workout plan: {str(e)}")
        # Return a 500 status for unexpected errors
        return JSONResponse({"error": "An error occurred. Please try again."}, status_code=500)

# -----------------------------------------------------------------------------
# Run the application
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)