
### README.md

```markdown
# Heart Attack Risk Assessment

This project aims to develop a machine learning model to predict the risk of heart attack based on various health metrics.
The project includes model development, preprocessing, hyperparameter tuning, and deployment using Streamlit and Flask.

## Table of Contents

1. [Dataset](#dataset)
2. [Environment Setup](#environment-setup)
3. [Model Development](#model-development)
4. [Deployment](#deployment)
   - [Streamlit](#streamlit)
   - [Flask](#flask)
5. [Docker Deployment](#docker-deployment)
6. [Usage](#usage)
7. [Contributing](#contributing)
8. [License](#license)

## Dataset

The dataset used in this project is the Heart Attack Prediction Dataset from Kaggle. You can download it from [here](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction).

### Sample Data Points

```plaintext
age,sex,total_cholesterol,ldl,hdl,systolic_bp,diastolic_bp,smoking,diabetes,heart_attack
57,1,229.46364245989878,175.87912925154933,39.22568705527789,124.07012716313945,91.37878025428563,0,0,0
58,1,186.46411960718476,128.9849160885468,34.95096762813148,95.49255176716945,64.35504019073372,1,0,0
37,1,251.30071919021645,152.34759243501477,45.91328769795121,99.51933468568974,64.95314701865036,0,1,0
55,1,192.05890823360733,116.80368391178928,67.20892502595552,122.46000218901882,73.8213817518664,0,0,0
```

## Environment Setup

### Prerequisites

- Python 3.12
- Docker (optional, for containerization)
- Pip

### Steps

1. **Clone the Repository**

   ```bash
   [git clone https://github.com/yourusername/heart-attack-risk-assessment.git](https://github.com/Blessman-Newton/Heart-Attack-Risk-Assessment)
   cd heart-attack-risk-assessment
   ```

2. **Create a Virtual Environment**

   ```bash
   python -m venv .venv
   ```

3. **Activate the Virtual Environment**

   - **Windows**

     ```bash
     .venv\Scripts\activate
     ```

   - **macOS/Linux**

     ```bash
     source .venv/bin/activate
     ```

4. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

5. **Create `.env` File**

   Create a `.env` file in the root directory with the following content:

   ```plaintext
   MODEL_PATH=./models
   PREPROCESSOR_PATH=./preprocessors
   ```

## Model Development

### Data Preprocessing

- **Numeric Features**: `Age`, `TotalCholesterol`, `LDL`, `HDL`, `SystolicBP`, `DiastolicBP`
- **Categorical Features**: `Sex`, `Smoking`, `Diabetes`

### Split Data

- The data is split into training and testing sets with an 80/20 ratio.

### Handle Multicollinearity

- Variance Inflation Factor (VIF) is calculated to identify and remove highly collinear features.

### Train and Evaluate Models

- **Random Forest Classifier**
- **Logistic Regression**
- **Multi-Layer Perceptron (MLP) Classifier**

### Hyperparameter Tuning

- `GridSearchCV` is used to find the best hyperparameters for each model.

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score

### Sensitivity Analysis

- **Permutation Importance**
- **Sobol Sensitivity Analysis**

## Deployment

### Streamlit

Streamlit is used to create a web-based interface for the model predictions.

#### Steps

1. **Run Streamlit App**

   ```bash
   streamlit run app.py
   ```

2. **Access the App**

   Open your browser and go to `http://localhost:8501`.

### Flask

Flask is used to create an API endpoint for the model predictions.

#### Steps

1. **Run Flask App**

   ```bash
   flask run --host=0.0.0.0 --port=5000
   ```

2. **Access the API**

   Use a tool like `curl` or Postman to send POST requests to `http://localhost:5000/predict`.

   **Example POST Request**

   ```bash
   curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{
       "Age": 57,
       "Sex": "Male",
       "TotalCholesterol": 229.46364245989878,
       "LDL": 175.87912925154933,
       "HDL": 39.22568705527789,
       "SystolicBP": 124.07012716313945,
       "DiastolicBP": 91.37878025428563,
       "Smoking": "No",
       "Diabetes": "No"
   }'
   ```

## Docker Deployment

Docker is used to containerize the application, making it easy to deploy and run consistently across different environments.

### Steps

1. **Build Docker Image**

   ```bash
   docker build -t heart-attack-risk-assessment .
   ```

2. **Run Docker Container**

   ```bash
   docker run -p 8501:8501 -p 5000:5000 heart-attack-risk-assessment
   ```

3. **Access the Applications**

   - **Streamlit**: `http://localhost:8501`
   - **Flask API**: `http://localhost:5000/predict`

## Usage

### Input Features

- **Age**: Age of the patient (integer)
- **Sex**: Gender of the patient (`Male` or `Female`)
- **TotalCholesterol**: Total cholesterol level (float)
- **LDL**: Low-density lipoprotein level (float)
- **HDL**: High-density lipoprotein level (float)
- **SystolicBP**: Systolic blood pressure (float)
- **DiastolicBP**: Diastolic blood pressure (float)
- **Smoking**: Smoking status (`Yes` or `No`)
- **Diabetes**: Diabetes status (`Yes` or `No`)

### Output

- **Random Forest Prediction**: `Heart Attack` or `No Heart Attack`
- **Logistic Regression Prediction**: `Heart Attack` or `No Heart Attack`
- **MLP Prediction**: `Heart Attack` or `No Heart Attack`

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a pull request.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgments

- [Kaggle](https://www.kaggle.com/) for providing the dataset.
- [Streamlit](https://streamlit.io/) for the web-based interface.
- [Flask](https://flask.palletsprojects.com/) for the API endpoint.
- [Docker](https://www.docker.com/) for containerization.

---

**Note**: Ensure that the dataset (`heart.csv`) is placed in the root directory of the project before running the scripts.

### Directory Structure

```
heart_attack_risk_assessment/
│
├── heart.csv
├── app.py
├── flask_app.py
├── Dockerfile
├── requirements.txt
├── preprocessor.pkl
├── best_rf_model.pkl
├── best_lr_model.pkl
├── best_mlp_model.pkl
└── .env
```

### Example Usage

#### Streamlit

1. **Run the Streamlit App**

   ```bash
   streamlit run app.py
   ```

2. **Access the App**

   Open your browser and go to `http://localhost:8501`.

#### Flask API

1. **Run the Flask App**

   ```bash
   flask run --host=0.0.0.0 --port=5000
   ```

2. **Send a POST Request**

   ```bash
   curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{
       "Age": 57,
       "Sex": "Male",
       "TotalCholesterol": 229.46364245989878,
       "LDL": 175.87912925154933,
       "HDL": 39.22568705527789,
       "SystolicBP": 124.07012716313945,
       "DiastolicBP": 91.37878025428563,
       "Smoking": "No",
       "Diabetes": "No"
   }'
   ```

### Docker Commands

1. **Build Docker Image**

   ```bash
   docker build -t heart-attack-risk-assessment .
   ```

2. **Run Docker Container**

   ```bash
   docker run -p 8501:8501 -p 5000:5000 heart-attack-risk-assessment
   ```

3. **Access the Applications**

   - **Streamlit**: `http://localhost:8501`
   - **Flask API**: `http://localhost:5000/predict`

---

Feel free to reach out if you have any questions or need further assistance!
```

### Additional Notes

- **Dataset**: Ensure that the `heart.csv` file is placed in the root directory of the project.
- **Environment Variables**: The `.env` file is used to specify paths for models and preprocessors. You can modify these paths as needed.
- **Docker**: The Dockerfile sets up the environment and installs the necessary dependencies. It exposes ports 8501 for Streamlit and 5000 for Flask.

This `README.md` file provides a comprehensive guide for setting up, training, and deploying your Heart Attack Risk Assessment application.
