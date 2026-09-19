# Student Performance Prediction Using Machine Learning

## Project Overview
https://student-2b543.streamlit.app/
This project predicts a student's test score using Machine Learning based on:

- Age
- Study Hours
- Attendance Percentage

It also predicts the student's grade using a classification model.

## Features

- Data loading using Pandas
- Missing value handling
- Duplicate value removal
- Data type checking
- Label Encoding
- Exploratory Data Analysis
- Regression models
- Classification model
- Model evaluation
- Student test score prediction
- Model saving using Joblib

## Machine Learning Models

### Regression

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

### Classification

- Random Forest Classifier

## Evaluation Metrics

Regression:

- R² Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)

Classification:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook

## Project Structure

```text
Student-Performance-Prediction/
│
├── data/
│   └── student_performance.csv
│
├── models/
│   ├── student_score_model.pkl
│   ├── student_scaler.pkl
│   └── student_grade_model.pkl
│
├── notebook/
│   └── student_performance.ipynb
│
├── screenshots/
│
├── src/
│
├── README.md
├── requirements.txt
└── .gitignore
