from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pydantic import BaseModel, EmailStr

from typing import List, Optional, Literal
import json
import math
from enum import Enum
from fastapi.middleware.cors import CORSMiddleware
import random
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
import bcrypt
import os
from dotenv import load_dotenv

load_dotenv()




app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to MongoDB
MONGO_URL = os.getenv('MONGO_URL')
client = AsyncIOMotorClient(MONGO_URL)
db = client.healthtrack
users_collection = db.users

# Import from our model file
from model import SymptomRequest, check_deficiency_with_contradictions, check_deficiency_simple


# Pydantic model for user signup/login
class UserSignup(BaseModel):
    fullName: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str


class Gender(str, Enum):
    male = "male"
    female = "female"
    
class ActivityLevel(str, Enum):
    sedentary = "sedentary"
    light = "light"
    moderate = "moderate"
    active = "active"
    very_active = "very_active"

class UserInput(BaseModel):
    age: int = Field(..., gt=0, lt=120)
    height: float = Field(..., gt=0)  # cm
    weight: float = Field(..., gt=0)  # kg
    gender: Gender
    activity_level: ActivityLevel
    weight_loss_goal: float = Field(..., ge=0, le=1)  # kg per week
    number_of_meals: int = Field(..., ge=1, le=6)

class Ingredient(BaseModel):
    name: str
    amount: str

class NutritionalValues(BaseModel):
    protein: float
    carbohydrates: float
    fat: float
    fiber: float

class Macronutrients(BaseModel):
    protein: float
    carbohydrates: float
    fat: float

class Meal(BaseModel):
    meal: str
    name: str
    calories: int
    nutritional_values: NutritionalValues
    ingredients: List[Ingredient]
    instructions: str

class NutritionSummary(BaseModel):
    total_calories: int
    macronutrients: Macronutrients

class MealPlanResponse(BaseModel):
    user_info: UserInput
    nutrition_summary: NutritionSummary
    meal_plan: List[Meal]

# Load meal plan database (in a real application, this would be more sophisticated)
with open("C:\\Users\\gvnba\\Downloads\\bmi\\meal-plans-db.json", "r") as f:
    meal_plans_db = json.load(f)

def calculate_bmr(user: UserInput) -> float:
    """Calculate Basal Metabolic Rate using Mifflin-St Jeor Equation"""
    if user.gender == Gender.male:
        return (10 * user.weight) + (6.25 * user.height) - (5 * user.age) + 5
    else:
        return (10 * user.weight) + (6.25 * user.height) - (5 * user.age) - 161

def calculate_tdee(bmr: float, activity_level: ActivityLevel) -> float:
    """Calculate Total Daily Energy Expenditure"""
    multipliers = {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "active": 1.725,
        "very_active": 1.9
    }
    return bmr * multipliers[activity_level]

def calculate_target_calories(tdee: float, weight_loss_goal: float) -> int:
    """Calculate target calories based on weight loss goal (deficit of ~500-1000 calories per day per 0.5-1kg/week)"""
    # Each 0.5kg of weight loss per week requires roughly a 500 calorie deficit per day
    calorie_deficit = weight_loss_goal * 1000
    target_calories = max(1200, tdee - calorie_deficit)  # Never go below 1200 calories
    return math.floor(target_calories)

def calculate_macros(target_calories: int) -> Macronutrients:
    """Calculate macronutrient distribution"""
    # Standard distribution: 25% protein, 45% carbs, 30% fat
    protein_calories = target_calories * 0.25
    carb_calories = target_calories * 0.45
    fat_calories = target_calories * 0.30
    
    # Convert to grams (protein: 4 cal/g, carbs: 4 cal/g, fat: 9 cal/g)
    protein_grams = math.floor(protein_calories / 4)
    carb_grams = math.floor(carb_calories / 4)
    fat_grams = math.floor(fat_calories / 9)
    
    return Macronutrients(
        protein=protein_grams,
        carbohydrates=carb_grams,
        fat=fat_grams
    )

# Load database from JSON
with open('C:\\Users\\gvnba\\Downloads\\bmi\\deficiency_database.json', 'r') as f:
    deficiency_db = json.load(f)

class SymptomRequest(BaseModel):
    symptoms: list

def get_meals_for_plan(target_calories: int, number_of_meals: int) -> List[Meal]:
    """Get appropriate meals from database based on calories and number of meals"""
    
    breakfast_options = [meal for meal in meal_plans_db["meals"] if meal["meal_type"] == "breakfast"]
    lunch_options = [meal for meal in meal_plans_db["meals"] if meal["meal_type"] == "lunch"]
    dinner_options = [meal for meal in meal_plans_db["meals"] if meal["meal_type"] == "dinner"]
    snack_options = [meal for meal in meal_plans_db["meals"] if meal["meal_type"] == "snack"]
    
    selected_meals = []
    
    # Always include breakfast, lunch, and dinner first (randomly chosen)
    if number_of_meals >= 3:
        selected_meals.append(random.choice(breakfast_options))
        selected_meals.append(random.choice(lunch_options))
        selected_meals.append(random.choice(dinner_options))
        
        # Add random snacks if user wants more meals
        for _ in range(number_of_meals - 3):
            if snack_options:  # check if snacks are available
                selected_meals.append(random.choice(snack_options))
    else:
        # If user asked for less than 3 meals, prioritize: breakfast → lunch → dinner
        meal_types = ["breakfast", "lunch", "dinner"]
        for i in range(number_of_meals):
            meal_type = meal_types[i]
            if meal_type == "breakfast" and breakfast_options:
                selected_meals.append(random.choice(breakfast_options))
            elif meal_type == "lunch" and lunch_options:
                selected_meals.append(random.choice(lunch_options))
            elif meal_type == "dinner" and dinner_options:
                selected_meals.append(random.choice(dinner_options))
    
    # Convert selected meals to Meal objects
    result = []
    for meal_data in selected_meals:
        meal = Meal(
            meal=meal_data["meal_type"],
            name=meal_data["name"],
            calories=meal_data["calories"],
            nutritional_values=NutritionalValues(
                protein=meal_data["nutritional_values"]["protein"],
                carbohydrates=meal_data["nutritional_values"]["carbohydrates"],
                fat=meal_data["nutritional_values"]["fat"],
                fiber=meal_data["nutritional_values"]["fiber"]
            ),
            ingredients=[Ingredient(name=ing["name"], amount=ing["amount"]) for ing in meal_data["ingredients"]],
            instructions=meal_data["instructions"]
        )
        result.append(meal)
    
    return result


@app.post("/generate-meal-plan/", response_model=MealPlanResponse)
async def generate_meal_plan(user_input: UserInput):
    try:
        # Calculate BMR (Basal Metabolic Rate)
        bmr = calculate_bmr(user_input)
        
        # Calculate TDEE (Total Daily Energy Expenditure)
        tdee = calculate_tdee(bmr, user_input.activity_level)
        
        # Calculate target calories based on weight loss goal
        target_calories = calculate_target_calories(tdee, user_input.weight_loss_goal)
        
        # Calculate macronutrient distribution
        macros = calculate_macros(target_calories)
        
        # Get meals for the plan
        meals = get_meals_for_plan(target_calories, user_input.number_of_meals)
        
        # Create and return the meal plan response
        meal_plan = MealPlanResponse(
            user_info=user_input,
            nutrition_summary=NutritionSummary(
                total_calories=target_calories,
                macronutrients=macros
            ),
            meal_plan=meals
        )
        
        return meal_plan
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/check-deficiency/")
def api_check_deficiency(request: SymptomRequest):
    """
    Check for nutrient deficiencies based on symptoms with detailed analysis
    """
    return check_deficiency_with_contradictions(request.symptoms)

@app.post("/check-deficiency-simple/")
def api_check_deficiency_simple(request: SymptomRequest):
    """
    Simple deficiency check API for backward compatibility
    """
    return check_deficiency_simple(request.symptoms)

# Add OPTIONS handlers for the endpoints to support CORS preflight requests
@app.options("/check-deficiency/")
@app.options("/check-deficiency-simple/")
async def options_handler():
    """
    Handle OPTIONS requests for CORS preflight
    """
    return {"detail": "OK"}



# Signup Route
@app.post("/signup")
async def signup(user: UserSignup):
    existing_user = await users_collection.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Hash the password
    hashed_password = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt())
    
    user_doc = {
        "fullName": user.fullName,
        "email": user.email,
        "password": hashed_password,
    }
    
    result = await users_collection.insert_one(user_doc)
    return {"message": "User registered successfully", "user_id": str(result.inserted_id)}


# Login Route
@app.post("/login")
async def login(user: UserLogin):
    existing_user = await users_collection.find_one({"email": user.email})
    
    if not existing_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Check password
    if not bcrypt.checkpw(user.password.encode('utf-8'), existing_user['password']):
        raise HTTPException(status_code=401, detail="Incorrect password")
    
    return {"message": "Login successful", "fullName": existing_user["fullName"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)
