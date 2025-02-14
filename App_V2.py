import streamlit as st
import pandas as pd
import joblib
import numpy as np
from PIL import Image

# Load the logo and background
logo = Image.open("Pixel_Pulse_Studios_Logo.png")  # Replace with correct file path
angry_birds_icon = Image.open("angry_birds_icon.png")  # Replace with correct file path
background_image = "angry_birds_background.jpg"  # Replace with correct file path

# Set page configuration
st.set_page_config(page_title="Game Level Predictor", page_icon=angry_birds_icon, layout="wide")

# Custom CSS for styling
st.markdown(
    f"""
    <style>
    .stApp {{
        background: url('{background_image}') no-repeat center center fixed;
        background-size: cover;
    }}
    .title {{
        color: #ffcc00;
        text-align: center;
        font-size: 50px;
        font-weight: bold;
    }}
    .prediction-box {{
        padding: 15px;
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        color: #333333;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Load models
def load_model(filename):
    try:
        return joblib.load(filename)
    except Exception as e:
        st.error(f"Failed to load model {filename}: {e}")
        return None

models = {
    "Decision Tree": load_model("Decision_Tree.pkl"),
    "Linear Regression": load_model("Linear_Regression.pkl"),
    "Random Forest": load_model("Random_Forest.pkl")
}

# Remove models that failed to load
models = {name: model for name, model in models.items() if model is not None}

# Display logo
st.image(logo, width=400)

# Title
st.markdown("<div class='title'>Game Level Predictor App 🤖</div>", unsafe_allow_html=True)

st.write("🎮 Predict the next game level and difficulty using AI models! 🎮")

# Sidebar inputs
st.sidebar.header("Input Features 🎯")
level_completed = st.sidebar.slider("Last Level Attempted", 1, 10, 5)
last_level_attempts = st.sidebar.slider("Last Level Attempts", 1, 10, 5)
last_level_cleared = st.sidebar.radio("Last Level Cleared", ["Yes", "No"]) == "Yes"
difficulty = st.sidebar.slider("Difficulty", 1, 6, 3)

# Model selection
model_name = st.sidebar.selectbox("Select a Model", list(models.keys()))

# Feature sets
feature_sets = {
    "Decision Tree": ["last_level_attempts", "last_level_cleared", "difficulty"],
    "Linear Regression": ["last_level_attempts", "last_level_cleared", "difficulty"],
    "Random Forest": ["last_level_attempts", "last_level_cleared", "difficulty"]
}

# Input data dictionary
input_data = {
    "last_level_attempts": last_level_attempts,
    "last_level_cleared": int(last_level_cleared),
    "difficulty": difficulty,
    "level_completed": level_completed
}

# Recommendation logic
def recommend_nlod(last_level_attempts, last_level_cleared, difficulty, level_completed):
    if last_level_attempts > 5 and not last_level_cleared:
        return level_completed, max(1, difficulty - 1)
    elif last_level_cleared:
        return level_completed + 1, min(6, difficulty + 1)
    return level_completed, difficulty

if st.button("Predict Next Level and Difficulty 🕹️"):
    next_level, next_difficulty = recommend_nlod(last_level_attempts, last_level_cleared, difficulty, level_completed)
    
    st.markdown(
        f"""
        <div class='prediction-box'>
            Recommended Next Level: {next_level}<br>
            Next Level Of Difficulty (NLOD): {next_difficulty}
        </div>
        """,
        unsafe_allow_html=True
    )
    
    model = models.get(model_name)
    if model:
        selected_features = feature_sets[model_name]
        input_features = np.array([[input_data[feature] for feature in selected_features]])
        try:
            prediction = model.predict(input_features)
            #st.write(f"**{model_name} Model Prediction:** Next Level: {int(prediction[0])}")
        except Exception as e:
            st.error(f"Prediction Error: {e}")
