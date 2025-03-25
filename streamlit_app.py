import streamlit as st
import joblib
import numpy as np

# Load the trained model and scaler
model_path = r"C:/Users/Dell/churn_project/churn_app/models/churn_model.pkl"
scaler_path = r"C:/Users/Dell/churn_project/churn_app/models/scaler.pkl"

# Ensure model files exist
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Streamlit UI
st.title("Bank Customer Churn Prediction")

# User inputs
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
balance = st.number_input("Balance", min_value=0.0, max_value=1000000.0, value=50000.0)
num_of_products = st.slider("Number of Products", min_value=1, max_value=4, value=1)
is_active_member = st.radio("Is Active Member?", [1, 0])

# Prepare input for model
features = np.array([[credit_score, age, balance, num_of_products, is_active_member]])
features_scaled = scaler.transform(features)  # Scale the input

# Predict
if st.button("Predict Churn"):
    prediction = model.predict(features_scaled)
    result = "Customer is likely to churn" if prediction[0] == 1 else "Customer is not likely to churn"
    st.success(result)
