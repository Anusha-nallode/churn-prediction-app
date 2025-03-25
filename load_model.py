import joblib

try:
    model_path = "C:/Users/Dell/churn_project/churn_app/models/churn_model.pkl"
    model = joblib.load(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
