from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import joblib

# Load the preprocessor
preprocessor = joblib.load('preprocessor.pkl')

# Load the models
best_rf_model = joblib.load('best_rf_model.pkl')
best_lr_model = joblib.load('best_lr_model.pkl')
best_mlp_model = joblib.load('best_mlp_model.pkl')

# Define the preprocessing steps
numeric_features = ['age', 'total_cholesterol', 'ldl', 'hdl', 'systolic_bp', 'diastolic_bp']
categorical_features = ['sex', 'smoking', 'diabetes']

# Function to predict using the models
def predict(input_data):
    # Convert input data to a DataFrame
    input_df = pd.DataFrame([input_data], columns=numeric_features + categorical_features)
    
    # Preprocess the input data
    input_data_preprocessed = preprocessor.transform(input_df)
    
    # Make predictions with the best models
    pred_rf = best_rf_model.predict(input_data_preprocessed)
    pred_lr = best_lr_model.predict(input_data_preprocessed)
    pred_mlp = best_mlp_model.predict(input_data_preprocessed)
    
    return pred_rf[0], pred_lr[0], pred_mlp[0]

# Initialize the Flask app
app = Flask(__name__)

# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.json
    input_data = [
        data['Age'],
        data['Sex'],
        data['TotalCholesterol'],
        data['LDL'],
        data['HDL'],
        data['SystolicBP'],
        data['DiastolicBP'],
        data['Smoking'],
        data['Diabetes']
    ]
    pred_rf, pred_lr, pred_mlp = predict(input_data)
    return jsonify({
        'Random Forest Prediction': 'Heart Attack' if pred_rf == 1 else 'No Heart Attack',
        'Logistic Regression Prediction': 'Heart Attack' if pred_lr == 1 else 'No Heart Attack',
        'MLP Prediction': 'Heart Attack' if pred_mlp == 1 else 'No Heart Attack'
    })

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)