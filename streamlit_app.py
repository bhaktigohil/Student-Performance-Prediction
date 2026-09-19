import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "student_score_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "student_scaler.pkl"

# --------------------------------------------------
# Load trained model and scaler
# --------------------------------------------------

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


try:
    model, scaler = load_model()

except Exception as e:
    st.error("❌ Unable to load the trained model.")
    st.write("Check that these files exist:")
    st.code(
        "models/student_score_model.pkl\n"
        "models/student_scaler.pkl"
    )
    st.error(str(e))
    st.stop()

# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🎓 Student Performance Prediction")

st.write(
    "Predict an expected test score using "
    "age, study hours, and attendance."
)

st.divider()

# --------------------------------------------------
# User inputs
# --------------------------------------------------

age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=21,
    step=1
)

study_hours = st.number_input(
    "Study hours per day",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5
)

attendance = st.number_input(
    "Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=85.0,
    step=1.0
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict", type="primary"):

    input_data = pd.DataFrame({
        "Age": [age],
        "Study_Hours": [study_hours],
        "Attendance(%)": [attendance]
    })

    try:

        # Apply the same scaler used during training
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0]

        # Keep score within normal test-score range
        prediction = max(0, min(100, prediction))

        st.success(
            f"📊 Predicted Test Score: **{prediction:.2f}**"
        )

    except Exception as e:

        st.error("❌ Prediction failed.")
        st.code(str(e))

# --------------------------------------------------
# Information
# --------------------------------------------------

st.divider()

st.caption(
    "This prediction is an estimate based on the training dataset "
    "and should not be treated as a guaranteed result."
)