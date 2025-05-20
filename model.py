from pydantic import BaseModel
from typing import List, Dict, Set, Optional
import json
from sklearn import tree
import numpy as np
from collections import Counter

class BeautifulContradiction(BaseModel):
    contradiction_type: str
    description: str
    resolution_approach: str

class SymptomRequest(BaseModel):
    symptoms: list

# Load database
with open('deficiency_database.json', 'r') as f:
    deficiency_db = json.load(f)

# Create a mapping of symptoms to indices
all_symptoms = list(deficiency_db.keys())
symptom_to_idx = {symptom: idx for idx, symptom in enumerate(all_symptoms)}

# Create a mapping of deficiencies and foods to indices
all_deficiencies = set()
all_foods = set() 
for symptom_data in deficiency_db.values():
    all_deficiencies.update(symptom_data.get("deficiencies", []))
    all_foods.update(symptom_data.get("foods", []))

all_deficiencies = list(all_deficiencies)
all_foods = list(all_foods)
deficiency_to_idx = {deficiency: idx for idx, deficiency in enumerate(all_deficiencies)}
food_to_idx = {food: idx for idx, food in enumerate(all_foods)}

# Build training data for the decision tree
X = []  # Features (symptoms)
y_deficiencies = []  # Target (deficiencies)
y_foods = []  # Target (foods)

for symptom, data in deficiency_db.items():
    # Create one-hot encoded vector for this symptom
    symptom_vector = [0] * len(all_symptoms)
    symptom_vector[symptom_to_idx[symptom]] = 1
    
    # Create vectors for deficiencies and foods
    deficiency_vector = [0] * len(all_deficiencies)
    food_vector = [0] * len(all_foods)
    
    for deficiency in data.get("deficiencies", []):
        deficiency_vector[deficiency_to_idx[deficiency]] = 1
    
    for food in data.get("foods", []):
        food_vector[food_to_idx[food]] = 1
    
    X.append(symptom_vector)
    y_deficiencies.append(deficiency_vector)
    y_foods.append(food_vector)

# Convert to numpy arrays
X = np.array(X)
y_deficiencies = np.array(y_deficiencies)
y_foods = np.array(y_foods)

# Train decision trees (one for each deficiency and food)
deficiency_models = []
for i in range(len(all_deficiencies)):
    clf = tree.DecisionTreeClassifier(max_depth=5)
    clf.fit(X, y_deficiencies[:, i])
    deficiency_models.append(clf)

food_models = []
for i in range(len(all_foods)):
    clf = tree.DecisionTreeClassifier(max_depth=5)
    clf.fit(X, y_foods[:, i])
    food_models.append(clf)

# Function to identify beautiful contradictions
def identify_contradictions(symptoms, deficiencies, foods):
    contradictions = []
    
    # Check for nutrient competition (e.g., calcium and iron)
    competing_pairs = [
        ("Iron", "Calcium"),  # Calcium inhibits iron absorption
        ("Zinc", "Copper"),   # Zinc can inhibit copper absorption
        ("Vitamin A", "Vitamin D"),  # Can compete for absorption
    ]
    
    for pair in competing_pairs:
        if pair[0] in deficiencies and pair[1] in deficiencies:
            contradictions.append(BeautifulContradiction(
                contradiction_type="Nutrient Competition",
                description=f"{pair[0]} and {pair[1]} may compete for absorption",
                resolution_approach=f"Space out foods high in {pair[0]} and {pair[1]} by 2 hours"
            ))
    
    # Check for symptoms that point to opposing remedies
    symptom_pairs = [
        ("fatigue and weakness", "mental health issues"),  # May need both stimulating and calming foods
        ("bone pain", "skin problems")  # May need both calcium and vitamin A which can counteract
    ]
    
    for pair in symptom_pairs:
        if pair[0] in symptoms and pair[1] in symptoms:
            contradictions.append(BeautifulContradiction(
                contradiction_type="Symptom Conflict",
                description=f"Symptoms '{pair[0]}' and '{pair[1]}' may need opposing remedies",
                resolution_approach="Balance intake and consider timing of different foods"
            ))
    
    # Check for conflicting food recommendations
    food_categories = {
        "High Protein": ["Red meat", "Chicken", "Fish", "Eggs"],
        "Plant-Based": ["Lentils", "Spinach", "Nuts", "Seeds"],
        "Dairy": ["Milk", "Cheese", "Yogurt"]
    }
    
    found_categories = set()
    for food in foods:
        for category, items in food_categories.items():
            if food in items:
                found_categories.add(category)
    
    if "High Protein" in found_categories and "Plant-Based" in found_categories:
        contradictions.append(BeautifulContradiction(
            contradiction_type="Dietary Pattern Conflict",
            description="Both animal and plant sources recommended",
            resolution_approach="Consider a flexitarian approach with both sources"
        ))
    
    return contradictions

# Enhanced prediction function
def predict_with_decision_tree(symptoms):
    # Convert symptoms to feature vector
    symptom_vector = [0] * len(all_symptoms)
    for symptom in symptoms:
        symptom_lower = symptom.lower()
        # Handle similar but not exact matches
        closest_match = symptom_lower
        if symptom_lower not in symptom_to_idx:
            for known_symptom in all_symptoms:
                if symptom_lower in known_symptom or known_symptom in symptom_lower:
                    closest_match = known_symptom
                    break
        
        if closest_match in symptom_to_idx:
            symptom_vector[symptom_to_idx[closest_match]] = 1
    
    # Predict deficiencies and foods
    deficiency_predictions = []
    for idx, model in enumerate(deficiency_models):
        if model.predict([symptom_vector])[0] == 1:
            deficiency_predictions.append(all_deficiencies[idx])
    
    food_predictions = []
    for idx, model in enumerate(food_models):
        if model.predict([symptom_vector])[0] == 1:
            food_predictions.append(all_foods[idx])
    
    # Fallback to original database method if decision tree gives no results
    if not deficiency_predictions or not food_predictions:
        deficiencies = set()
        recommended_foods = set()
        
        for symptom in symptoms:
            symptom_lower = symptom.lower()
            if symptom_lower in deficiency_db:
                data = deficiency_db[symptom_lower]
                deficiencies.update(data.get("deficiencies", []))
                recommended_foods.update(data.get("foods", []))
        
        deficiency_predictions = list(deficiencies) if not deficiency_predictions else deficiency_predictions
        food_predictions = list(recommended_foods) if not food_predictions else food_predictions
    
    # Identify and handle contradictions
    contradictions = identify_contradictions(symptoms, deficiency_predictions, food_predictions)
    
    return deficiency_predictions, food_predictions, contradictions

# Function to handle deficiency checking with detailed analysis
def check_deficiency_with_contradictions(symptoms):
    deficiencies, foods, contradictions = predict_with_decision_tree(symptoms)
    
    # Format contradictions for response
    contradiction_info = [
        {
            "type": c.contradiction_type,
            "description": c.description,
            "resolution": c.resolution_approach
        }
        for c in contradictions
    ]
    
    # Add weighted importance score for deficiencies based on frequency across symptoms
    deficiency_counter = Counter()
    for symptom in symptoms:
        symptom_lower = symptom.lower()
        if symptom_lower in deficiency_db:
            for deficiency in deficiency_db[symptom_lower].get("deficiencies", []):
                deficiency_counter[deficiency] += 1
    
    # Create priority recommendations
    deficiency_priorities = [
        {
            "deficiency": deficiency,
            "priority": count / len(symptoms),
            "symptoms_related": [
                symptom for symptom in symptoms 
                if symptom.lower() in deficiency_db and 
                deficiency in deficiency_db[symptom.lower()].get("deficiencies", [])
            ]
        }
        for deficiency, count in deficiency_counter.items()
    ]
    
    # Sort by priority
    deficiency_priorities.sort(key=lambda x: x["priority"], reverse=True)
    
    return {
        "deficiencies": deficiencies,
        "recommended_foods": foods,
        "contradictions": contradiction_info,
        "deficiency_priorities": deficiency_priorities
    }

# Simple deficiency check function for backward compatibility
def check_deficiency_simple(symptoms):
    deficiencies = set()
    recommended_foods = set()

    for symptom in symptoms:
        symptom_lower = symptom.lower()
        if symptom_lower in deficiency_db:
            data = deficiency_db[symptom_lower]
            deficiencies.update(data.get("deficiencies", []))
            recommended_foods.update(data.get("foods", []))

    return {
        "deficiencies": list(deficiencies),
        "recommended_foods": list(recommended_foods)
    }
