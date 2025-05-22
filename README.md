# 🩺 HealthMate - Backend

The **HealthMate** backend is a **FastAPI**-powered service that provides smart health advisory features such as **BMI calculation**, **personalized meal planning**, and **nutritional deficiency detection**. It integrates **machine learning algorithms** like K-Means and Decision Tree to deliver personalized and data-driven health insights.

---

## 🚀 Features

- 🧮 **BMI Calculation** based on user-provided height and weight
- 🍱 **Meal Plan Recommendation** using **K-Means Clustering** based on user age, weight, height, and fitness goals (e.g., weight loss)
- 🌾 **Deficiency Checker** using a **Decision Tree Classifier** based on symptoms
- 👤 **User Signup & Login** functionality
- 📈 **API-first Architecture** for smooth frontend integration

---

## 🛠️ Tech Stack

- **FastAPI** – Web framework for building APIs
- **Python Libraries**:
  - `scikit-learn` – For K-Means and Decision Tree models
  - `pymongo` – MongoDB integration
  - `pydantic` – Data validation and serialization
- **MongoDB** – For storing user data and logs
- **Uvicorn** – ASGI server for FastAPI

---

## 📡 API Endpoints

| Endpoint                | Method | Description                                                   |
|-------------------------|--------|---------------------------------------------------------------|
| `/signup`              | POST   | Register a new user with email and password                |
| `/login`               | POST   | Log in a user using email and password                              |
| `/generate-meal-plan/` | POST   | Recommend a meal plan using K-Means based on user input       |
| `/check-deficiency/`   | POST   | Predict deficiencies and suggest food using Decision Tree     |

---

📦 Related Repository
🔗 Frontend Repo: https://github.com/Sivajothi015/HEALTHMATE-FRONTEND
