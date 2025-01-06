# PDF Chatbot - FastAPI Application with LangChain Integration

This repository hosts a FastAPI application that integrates LangChain for advanced question answering and document retrieval capabilities. The application facilitates the uploading of PDF files, extracts their text, and enables querying for answers based on user input, leveraging document content.

## Features

- **Upload PDF**: Allows for the uploading of PDF files to the server. These files are then processed and their content is stored in a vector store for quick retrieval.
- **WebSocket Chat**: Enables real-time question answering via a WebSocket endpoint, providing immediate responses to user queries.
- **Document Retrieval**: Capable of retrieving and displaying source documents that relate to the questions asked, aiding in transparency and verifiability of answers.
- **Memory Integration**: Utilizes a conversation history buffer to enhance the relevance and accuracy of responses based on previous interactions.

## Setup Instructions

To deploy and run this application locally, please follow the steps outlined below:

### Prerequisites

- Python 3.10 or newer
- Required Python packages, which can be installed via pip using the provided `requirements.txt` file.

### Running the Application

1. **Install dependencies**:
   Ensure that all required packages are installed by running:

   ```bash
   pip install -r requirements.txt
   ```

2. **Start the FastAPI server**:
   Launch the application server using the following command:

   ```bash
   uvicorn main:app --host 127.0.0.1 --port 8000
   ```


3. **Access the API**:
   After starting the server, the API will be accessible via `http://localhost:8000` in your web browser.

### Endpoints

- **GET `/get_documents/`**: Retrieves a list of all documents that have been uploaded and processed.
- **POST `/upload_pdf/`**: Endpoint for uploading a PDF file to be processed and added to the document store.
- **WebSocket `/ws/chat`**: WebSocket endpoint for engaging in real-time, interactive question answering sessions.

### Example Usage

1. **Upload a PDF**:
   Utilize the `/upload_pdf/` endpoint to upload a document.

2. **Establish WebSocket Connection**:
   Connect to the `/ws/chat` endpoint using a WebSocket client.

3. **Query the System**:
   Send your questions through the WebSocket connection to receive real-time answers based on the uploaded and processed PDF content.

## Technologies Used

- **FastAPI**: A modern, fast web framework for building APIs with Python 3.7+.
- **LangChain**: A comprehensive library for natural language processing, focusing on question answering and document retrieval.
- **Chroma Vector Store**: Implements a persistent storage solution for document embeddings, facilitating efficient retrieval.
- **PyPDF2**: A Python library used for handling PDF file manipulations.