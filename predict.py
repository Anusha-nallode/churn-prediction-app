import pandas as pd
import joblib
import os
import numpy as np

# Set paths
model_dir = r"C:/Users/Dell/churn_project/churn_app/models"
model_path = os.path.join(model_dir, "churn_model.pkl")
scaler_path = os.path.join(model_dir, "scaler.pkl")

# Load the trained model and scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Function to predict churn
def predict_churn(new_data):
    # Convert new data to DataFrame
    df = pd.DataFrame([new_data])

    # Convert categorical values if present
    if 'PhoneService' in df.columns:
        df['PhoneService'] = df['PhoneService'].map({'Yes': 1, 'No': 0}).fillna(0)
    
    # Convert 'TotalCharges' to numeric
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

    # Select required features
    features = ['SeniorCitizen', 'tenure', 'PhoneService', 'MonthlyCharges', 'TotalCharges']
    df = df[features]

    # Scale the input data
    df_scaled = scaler.transform(df)

    # Predict
    prediction = model.predict(df_scaled)
    probability = model.predict_proba(df_scaled)[:, 1]  # Probability of churn

    # Return results
    return {"Prediction": "Churn" if prediction[0] == 1 else "No Churn", "Probability": probability[0]}

# Example customer input
new_customer = {
    "SeniorCitizen": 1,
    "tenure": 24,
    "PhoneService": "Yes",
    "MonthlyCharges": 70.0,
    "TotalCharges": 1680.0
}

# Make prediction
result = predict_churn(new_customer)
print("🔹 Prediction:", result["Prediction"])
print("🔹 Churn Probability:", round(result["Probability"] * 100, 2), "%")
