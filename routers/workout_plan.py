# backend/routers/workout_plan.py

from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from workout_plan_logic import generate_workout_plan_with_ai, adjust_workout_plan_with_ai

router = APIRouter()
templates = Jinja2Templates(directory="templates")

@router.get("/workout_planner", response_class=HTMLResponse)
async def workout_planner_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@router.post("/generate_workout_plan")
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
    user_data = {
        "name": name,
        "age": age,
        "gender": gender,
        "height": height,
        "weight": weight,
        "goal": goal,
        "activity_level": activity_level,
        "workout_type": workout_type,
        "experience_level": experience_level,
        "equipment": equipment,
        "time_available": time_available,
    }
    
    workout_plan = generate_workout_plan_with_ai(user_data)
    
    print("DEBUG: Generated plan:", workout_plan)
    
    return JSONResponse(
        content={
            "plan": workout_plan,
            "name": name
        }
    )

@router.post("/adjust_workout_plan")
async def adjust_workout_plan(
    request: Request,
    name: str = Form(...),
    daily_diet: str = Form(...),
    daily_sleep: str = Form(...)
):
    adjusted_plan = adjust_workout_plan_with_ai(name, daily_diet, daily_sleep)

    return JSONResponse(
        content={
            "adjusted_plan": adjusted_plan,
            "name": name
        }
    )