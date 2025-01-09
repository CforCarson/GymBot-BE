# workout_plan_logic.py

import json
from openai import OpenAI
from typing import Dict, List

# Configure OpenAI client
client = OpenAI(
    api_key="sk-RSlhENXN4JW0Hq5v060fDaB48c0d4888B402Ea723949300c", 
    base_url="https://api.gpt.ge/v1/"  # Endpoint URL
)

def generate_workout_plan_with_ai(user_data: Dict) -> Dict[str, List[str]]:
    # Create a prompt for the AI
    prompt = create_ai_prompt(user_data)
    
    try:
        # Make API call using the new OpenAI client
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional fitness trainer. Generate a detailed weekly workout plan based on "
                        "the user's data. Provide the plan in JSON format with days of the week as keys and lists "
                        "of exercises as values."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        # Parse the AI response
        response_content = completion.choices[0].message.content.strip()
        try:
            workout_plan = json.loads(response_content)
        except json.JSONDecodeError:
            # Attempt to extract JSON from the response
            workout_plan = extract_json_from_response(response_content)
            if workout_plan is None:
                # Fallback to default plan if AI response is not valid JSON
                return generate_default_plan()

        return workout_plan

    except Exception as e:
        print(f"Error generating workout plan: {str(e)}")
        return generate_default_plan()

def create_ai_prompt(user_data: Dict) -> str:
    # Convert goal, workout type, and equipment numbers to descriptions
    goals = {1: "Weight Loss", 2: "Muscle Gain", 3: "Maintenance"}
    workout_types = {1: "Cardio", 2: "Strength Training", 3: "Flexibility", 4: "HIIT", 5: "Mixed"}
    equipment_types = {1: "No Equipment", 2: "Basic Equipment", 3: "Full Gym Access"}
    experience_levels = {1: "Beginner", 2: "Intermediate", 3: "Advanced"}
    
    prompt = f"""
    Please create a detailed weekly workout plan for a person with the following characteristics:

    Personal Information:
    - Age: {user_data['age']}
    - Gender: {user_data['gender']}
    - Height: {user_data['height']} cm
    - Weight: {user_data['weight']} kg

    Fitness Parameters:
    - Goal: {goals.get(user_data['goal'], 'Unknown')}
    - Activity Level: Level {user_data['activity_level']}
    - Preferred Workout Type: {workout_types.get(user_data['workout_type'], 'Unknown')}
    - Experience Level: {experience_levels.get(user_data['experience_level'], 'Unknown')}
    - Available Equipment: {equipment_types.get(user_data['equipment'], 'Unknown')}
    - Time Available: {user_data['time_available']} minutes per day

    Please provide a detailed weekly plan with specific exercises, sets, reps, and rest periods where applicable.
    Return the response in JSON format with days of the week as keys and lists of exercises as values.
    """
    return prompt

def extract_json_from_response(response_content: str) -> Dict:
    # Attempt to find and extract JSON from the response content
    try:
        start_index = response_content.index('{')
        end_index = response_content.rindex('}') + 1
        json_str = response_content[start_index:end_index]
        return json.loads(json_str)
    except (ValueError, json.JSONDecodeError):
        return None

def generate_default_plan() -> Dict[str, List[str]]:
    """Generate a basic default plan in case of API failure"""
    return {
        'Monday': ['30 minutes walking', 'Basic stretching'],
        'Tuesday': ['Body weight exercises', 'Light cardio'],
        'Wednesday': ['Rest day', 'Light stretching'],
        'Thursday': ['30 minutes walking', 'Basic stretching'],
        'Friday': ['Body weight exercises', 'Light cardio'],
        'Saturday': ['Active recovery', 'Light walking'],
        'Sunday': ['Rest day', 'Light stretching']
    }

def adjust_workout_plan_with_ai(name, adjustment_data):
    """
    Adjust the existing workout plan based on the user's dynamic inputs
    (e.g., daily_diet, daily_sleep, injuries, stress_level, etc.).
    """

    # 1) Build an AI prompt referencing the new data
    #    This example does not pass the old plan. Instead, it re-creates
    #    instructions for a brand-new plan that includes user data + adjustments.
    #    In reality, consider passing an "existing_plan" so the AI can refine it.
    prompt = f"""
    The user has some additional data for adjusting their workout plan:
    {adjustment_data}

    Please create a detailed weekly workout plan in JSON format (days of week
    as keys, arrays of exercises/objects as values) that incorporates these adjustments.
    Use the same structure you normally return for a plan.
    """

    try:
        # 2) Make the OpenAI call
        #    We'll assume you already have a configured 'client' from openai import OpenAI
        #    or you can import openai directly as in your "generate_workout_plan_with_ai" function.
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional fitness trainer. Adjust an existing "
                        "workout plan based on new user input. Return valid JSON with "
                        "days of the week as keys and exercise details as values."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        # 3) Parse the AI response
        response_content = completion.choices[0].message.content.strip()
        try:
            adjusted_plan = json.loads(response_content)
        except json.JSONDecodeError:
            # Attempt to extract JSON from the response if AI doesn't return clean JSON
            adjusted_plan = extract_json_from_response(response_content)
            if adjusted_plan is None:
                # fallback if AI response is not valid JSON
                return generate_default_plan()

        return adjusted_plan

    except Exception as e:
        print(f"Error adjusting workout plan: {str(e)}")
        return generate_default_plan()