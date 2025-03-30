import joblib
import pandas as pd
import numpy as np
import os

# 🔄 Load trained model & encoders (Use absolute paths to avoid errors)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get chatbot.py directory
MODEL_PATH = os.path.join(BASE_DIR, "../models/xgboost_disease_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "../models/label_encoder.pkl")
MAPPING_PATH = os.path.join(BASE_DIR, "../models/disease_mapping.pkl")

# ✅ Load model and mappings
try:
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    disease_mapping = joblib.load(MAPPING_PATH)
    model_features = model.feature_names_in_
except FileNotFoundError:
    print("⚠ Error: Model files not found! Ensure models exist in the 'models' directory.")
    model = None
    label_encoder = None
    disease_mapping = {}

# ✅ Define keyword mappings for flexible input handling
YES_WORDS = {"yes", "y", "i have", "i do", "yeah", "sure"}
NO_WORDS = {"no", "n", "i don't", "not really", "nah"}
NONE_WORDS = {"none", "nothing", "no history", "not applicable", "n/a"}

BP_OPTIONS = {"low": "Low", "normal": "Normal", "high": "High"}
CHOL_OPTIONS = {"low": "Low", "normal": "Normal", "high": "High"}

GENDER_OPTIONS = {
    "male": "Male", "m": "Male", "man": "Male", "boy": "Male",
    "female": "Female", "f": "Female", "woman": "Female", "girl": "Female",
    "other": "Other", "non-binary": "Other", "nb": "Other", "trans": "Other"
}

# ✅ Function to process user input
def process_input(user_response, valid_options=None):
    user_response = user_response.lower().strip()

    # Handle Yes/No cases
    if any(word in user_response for word in YES_WORDS):
        return "Yes"
    if any(word in user_response for word in NO_WORDS):
        return "No"
    if any(word in user_response for word in NONE_WORDS):
        return "None"

    # Handle categorical inputs (Blood Pressure, Cholesterol, Gender)
    if valid_options:
        for key, value in valid_options.items():
            if key in user_response:
                return value

    return user_response.capitalize()

# ✅ Function to handle chatbot responses dynamically
def chatbot_response(user_message, step, user_data):
    if model is None:
        return "⚠ AI model not available. Please check server logs."

    try:
        if step == 0:
            return "👤 What is your name?"

        if step == 1:
            user_data["name"] = user_message.capitalize()
            return f"📅 Hi {user_data['name']}, how old are you?"

        if step == 2:
            if user_message.isdigit():
                user_data["Age"] = int(user_message)
                return "⚧ What is your gender? (Male/Female/Other)"
            else:
                return "⚠ Please enter a valid age."

        if step == 3:
            user_data["Gender"] = process_input(user_message, GENDER_OPTIONS)
            return "📖 Do you have any medical history? (Yes/No/None)"

        if step == 4:
            user_data["Medical History"] = process_input(user_message)
            return "🩸 What is your blood pressure? (Low/Normal/High)"

        if step == 5:
            user_data["Blood Pressure"] = process_input(user_message, BP_OPTIONS)
            return "🩺 What is your cholesterol level? (Low/Normal/High)"

        if step == 6:
            user_data["Cholesterol"] = process_input(user_message, CHOL_OPTIONS)
            return "🌡 Do you have a fever? (Yes/No)"

        if step == 7:
            user_data["Fever"] = 1 if process_input(user_message) == "Yes" else 0
            return "😷 Are you experiencing a cough? (Yes/No)"

        if step == 8:
            user_data["Cough"] = 1 if process_input(user_message) == "Yes" else 0
            return "💤 Are you feeling fatigued or weak? (Yes/No)"  # ✅ Added fatigue step

        if step == 9:
            user_data["Fatigue"] = 1 if process_input(user_message) == "Yes" else 0
            return "😮‍💨 Are you having trouble breathing? (Yes/No)"

        if step == 10:
            user_data["Difficulty Breathing"] = 1 if process_input(user_message) == "Yes" else 0

            # ✅ Prepare input data for model prediction
            input_data = {
                "Age": user_data.get("Age", 0),
                "Gender_Male": 1 if user_data.get("Gender") == "Male" else 0,
                "Gender_Female": 1 if user_data.get("Gender") == "Female" else 0,
                "Blood Pressure_Low": 1 if user_data.get("Blood Pressure") == "Low" else 0,
                "Blood Pressure_Normal": 1 if user_data.get("Blood Pressure") == "Normal" else 0,
                "Blood Pressure_High": 1 if user_data.get("Blood Pressure") == "High" else 0,
                "Cholesterol Level_Low": 1 if user_data.get("Cholesterol") == "Low" else 0,
                "Cholesterol Level_Normal": 1 if user_data.get("Cholesterol") == "Normal" else 0,
                "Cholesterol Level_High": 1 if user_data.get("Cholesterol") == "High" else 0,
                "Fever": user_data.get("Fever", 0),
                "Cough": user_data.get("Cough", 0),
                "Fatigue": user_data.get("Fatigue", 0),  # ✅ Added fatigue field
                "Difficulty Breathing": user_data.get("Difficulty Breathing", 0)
            }

            # ✅ Create DataFrame with aligned model features
            input_df = pd.DataFrame([input_data])
            for col in model_features:
                if col not in input_df:
                    input_df[col] = 0

            input_df = input_df[model_features]

            # ✅ Predict disease
            probs = model.predict_proba(input_df)[0]
            max_index = np.argmax(probs)
            predicted_disease = disease_mapping.get(max_index, "Unknown Disease")
            confidence = probs[max_index] * 100

            return (
                f"🩺 Predicted Disease: {predicted_disease}\n"
                f"📊 Confidence Score: {confidence:.2f}%\n\n"
                f"Would you like to start another diagnosis? (Yes/No)"
            )

        if step == 11:
            if process_input(user_message) == "Yes":
                step = 0  # Restart the diagnosis
                return "🔄 Restarting diagnosis...\n\n👤 Hellow once again !!! "
            else:
                step = -1  # ✅ End the session properly
                return "🙏 Thank you for using the AI Medical Assistant. Stay healthy! 😊"


    except Exception as e:
        return f"⚠ Error: {str(e)}"