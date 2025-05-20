import numpy as np
from sklearn.cluster import KMeans

def get_recommended_meals(target_nutrients, meal_plans_db, top_n=3, number_of_meals=3):
    """
    Recommends meals based on the K-Means algorithm, based on target nutritional values.
    
    :param target_nutrients: Target nutritional values (calories, protein, carbs, fat)
    :param meal_plans_db: Meal plans data containing details of meals
    :param top_n: Number of recommended meal combinations to return
    :param number_of_meals: Number of meals per day the user wants (1-6)
    :return: List of top N recommended meal combinations
    """
    # Define meal types based on number of meals requested
    meal_type_mapping = {
        1: ["breakfast"],
        2: ["breakfast", "dinner"],
        3: ["breakfast", "lunch", "dinner"],
        4: ["breakfast", "lunch", "dinner", "snack"],
        5: ["breakfast", "snack", "lunch", "dinner", "snack"],
        6: ["breakfast", "snack", "lunch", "snack", "dinner", "snack"]
    }
    
    requested_meal_types = meal_type_mapping.get(number_of_meals, ["breakfast", "lunch", "dinner"])
    
    # Group meals by type
    meals_by_type = {}
    for meal in meal_plans_db["meals"]:
        meal_type = meal["meal_type"]
        if meal_type not in meals_by_type:
            meals_by_type[meal_type] = []
        meals_by_type[meal_type].append(meal)
    
    # No need to handle special cases for snack types as we're only using "snack"
    
    # For each required meal type, find the best matching meal
    recommended_meal_plans = []
    
    # Track used meals across all plans to ensure diversity
    used_meals_by_type = {meal_type: [] for meal_type in requested_meal_types}
    
    # Create multiple meal plan combinations to return top_n options
    for plan_idx in range(top_n):
        meal_plan = {"meals": [], "total_nutrients": {"calories": 0, "protein": 0, "carbohydrates": 0, "fat": 0}}
        
        for meal_type in requested_meal_types:
            # Skip if meal type doesn't exist in our database
            if meal_type not in meals_by_type or not meals_by_type[meal_type]:
                continue
                
            # Get candidate meals for this meal type
            candidate_meals = meals_by_type[meal_type]
            
            # Prepare features for clustering
            meal_features = []
            for meal in candidate_meals:
                cals = meal["calories"] 
                prot = meal["nutritional_values"]["protein"]
                carbs = meal["nutritional_values"]["carbohydrates"]
                fat = meal["nutritional_values"]["fat"]
                meal_features.append([cals, prot, carbs, fat])
            
            # Apply K-Means clustering if we have enough meals
            if len(meal_features) >= 5:
                n_clusters = min(5, len(meal_features))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                kmeans.fit(meal_features)
                labels = kmeans.predict(meal_features)
                
                # Find the cluster closest to the target
                target_cluster = kmeans.predict([target_nutrients])[0]
                
                # Filter meals in the target cluster
                cluster_meals = [meal for i, meal in enumerate(candidate_meals) if labels[i] == target_cluster]
                cluster_features = [meal_features[i] for i in range(len(candidate_meals)) if labels[i] == target_cluster]
            else:
                # If not enough meals for clustering, use all meals
                cluster_meals = candidate_meals
                cluster_features = meal_features
            
            # Calculate distances to target
            distances = [np.linalg.norm(np.array(f) - target_nutrients) for f in cluster_features]
            
            # Create pairs of (distance, meal) and sort by distance
            distance_meal_pairs = list(zip(distances, cluster_meals))
            distance_meal_pairs.sort(key=lambda x: x[0])
            
            # Try to select a meal we haven't used before in this meal type
            selected_meal = None
            for _, meal_candidate in distance_meal_pairs:
                # Check if this meal has been used before for this meal type
                if meal_candidate["name"] not in used_meals_by_type[meal_type]:
                    selected_meal = meal_candidate
                    used_meals_by_type[meal_type].append(meal_candidate["name"])
                    break
            
            # If all meals have been used already, pick the best match
            if selected_meal is None and distance_meal_pairs:
                selected_meal = distance_meal_pairs[0][1]
                
            # If we found a meal, add it to the meal plan
            if selected_meal:
                meal_plan["meals"].append(selected_meal)
                meal_plan["total_nutrients"]["calories"] += selected_meal["calories"]
                meal_plan["total_nutrients"]["protein"] += selected_meal["nutritional_values"]["protein"]
                meal_plan["total_nutrients"]["carbohydrates"] += selected_meal["nutritional_values"]["carbohydrates"]
                meal_plan["total_nutrients"]["fat"] += selected_meal["nutritional_values"]["fat"]
        
        recommended_meal_plans.append(meal_plan)
    
    return recommended_meal_plans