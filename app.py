import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import joblib
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get paths from environment variables
model_path = os.getenv('MODEL_PATH', './models')
preprocessor_path = os.getenv('PREPROCESSOR_PATH', './preprocessors')

# Load the preprocessor
preprocessor = joblib.load(os.path.join(preprocessor_path, 'preprocessor.pkl'))

# Load the models
best_rf_model = joblib.load(os.path.join(model_path, 'best_rf_model.pkl'))
best_lr_model = joblib.load(os.path.join(model_path, 'best_lr_model.pkl'))
best_mlp_model = joblib.load(os.path.join(model_path, 'best_mlp_model.pkl'))

# Define the preprocessing steps
numeric_features = ['Age', 'TotalCholesterol', 'LDL', 'HDL', 'SystolicBP', 'DiastolicBP']
categorical_features = ['Sex', 'Smoking', 'Diabetes']

# Function to predict using the models
def predict(input_data):
    input_data_preprocessed = preprocessor.transform([input_data])
    pred_rf = best_rf_model.predict(input_data_preprocessed)
    pred_lr = best_lr_model.predict(input_data_preprocessed)
    pred_mlp = best_mlp_model.predict(input_data_preprocessed)
    return pred_rf[0], pred_lr[0], pred_mlp[0]

# Streamlit app
st.title('Heart Attack Risk Assessment')

# Input fields
age = st.number_input('Age', min_value=0, max_value=120, value=50)
sex = st.selectbox('Sex', ['Male', 'Female'])
total_cholesterol = st.number_input('Total Cholesterol (mg/dl)', min_value=0, max_value=600, value=200)
ldl = st.number_input('LDL (mg/dl)', min_value=0, max_value=300, value=120)
hdl = st.number_input('HDL (mg/dl)', min_value=0, max_value=100, value=40)
systolic_bp = st.number_input('Systolic BP (mm Hg)', min_value=0, max_value=200, value=120)
diastolic_bp = st.number_input('Diastolic BP (mm Hg)', min_value=0, max_value=120, value=80)
smoking = st.selectbox('Smoking', ['No', 'Yes'])
diabetes = st.selectbox('Diabetes', ['No', 'Yes'])

# Map categorical inputs to numerical values
sex_mapping = {'Male': 1, 'Female': 0}
smoking_mapping = {'Yes': 1, 'No': 0}
diabetes_mapping = {'Yes': 1, 'No': 0}

# Convert input data
input_data = [
    age,
    sex_mapping[sex],
    total_cholesterol,
    ldl,
    hdl,
    systolic_bp,
    diastolic_bp,
    smoking_mapping[smoking],
    diabetes_mapping[diabetes]
]

# Button to make prediction
if st.button('Predict'):
    pred_rf, pred_lr, pred_mlp = predict(input_data)
    
    st.write(f"Random Forest Prediction: {'Heart Attack' if pred_rf == 1 else 'No Heart Attack'}")
    st.write(f"Logistic Regression Prediction: {'Heart Attack' if pred_lr == 1 else 'No Heart Attack'}")
    st.write(f"MLP Prediction: {'Heart Attack' if pred_mlp == 1 else 'No Heart Attack'}")