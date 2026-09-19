from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "student_score_model.pkl"

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered",
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def grade_from_score(score: float) -> str:
    if score >= 91:
        return "A+"
    if score >= 81:
        return "A"
    if score >= 66:
        return "B"
    if score >= 51:
        return "C"
    return "F"

st.title("🎓 Student Performance Prediction")
st.write("Predict an expected test score from age, study hours, and attendance.")

with st.form("prediction_form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=21, step=1)
    study_hours = st.number_input("Study hours per day", min_value=0.0, max_value=24.0, value=5.0, step=0.1)
    attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=85.0, step=0.1)
    submitted = st.form_submit_button("Predict")

if submitted:
    model = load_model()
    features = pd.DataFrame([{
        "Age": age,
        "Study_Hours": study_hours,
        "Attendance(%)": attendance,
    }])
    score = float(model.predict(features)[0])
    score = max(0.0, min(100.0, score))
    grade = grade_from_score(score)

    st.success(f"Predicted test score: **{score:.1f}/100**")
    st.info(f"Estimated grade: **{grade}**")

st.caption("This prediction is an estimate based on the training dataset; it should not be treated as a guaranteed result.")