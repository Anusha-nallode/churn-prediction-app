import streamlit as st
import pickle
import numpy as np

# Function to load the pre-trained model
def load_model(model_name):
    with open(f"{model_name}.pkl", "rb") as file:
        model = pickle.load(file)
    return model

# Function to handle user inputs for churn prediction
def churn_page():
    # Sidebar navigation
    st.sidebar.title("Select a Model and Predict the Future 🔮")
    model_option = st.sidebar.selectbox("Select Model", ("logistic_regression", "random_forest"))
    
    # Displaying selected model
    st.sidebar.write(f"Selected model is: {model_option}")
    
    # Demographics inputs
    st.subheader("Demographics")
    gender = st.selectbox("Gender", ("Male", "Female"))
    senior_citizen = st.radio("Senior Citizen", ("No", "Yes"))
    partner = st.radio("Partner", ("Yes", "No"))
    dependents = st.radio("Dependents", ("Yes", "No"))
    
    # Basic services inputs
    st.subheader("Basic Services")
    phone_service = st.radio("Phone Service", ("Yes", "No"))
    multiple_lines = st.radio("Multiple Lines", ("Yes", "No"))
    internet_service = st.selectbox("Internet Service", ("Fiber optic", "DSL", "No"))
    online_security = st.radio("Online Security", ("Yes", "No"))
    
    # Other services inputs
    st.subheader("Other Services")
    online_backup = st.radio("Online Backup", ("Yes", "No"))
    device_protection = st.radio("Device Protection", ("Yes", "No"))
    tech_support = st.radio("Tech Support", ("Yes", "No"))
    streaming_tv = st.radio("Streaming TV", ("Yes", "No"))
    streaming_movies = st.radio("Streaming Movies", ("Yes", "No"))
    
    # Billing information
    st.subheader("Billing")
    contract = st.selectbox("Contract", ("Month-to-month", "Two year", "One year"))
    paperless_billing = st.radio("Paperless Billing", ("Yes", "No"))
    payment_method = st.selectbox("Payment Method", ("Electronic check", "Credit card (automatic)", "Mailed check", "Bank transfer (automatic)"))
    
    # Financial data
    st.subheader("Financial Data")
    tenure = st.number_input("Tenure (months)", min_value=1, max_value=71)
    monthly_charges = st.number_input("Monthly Charges", min_value=1, max_value=100)
    total_charges = st.number_input("Total Charges", min_value=1, max_value=100)

    # Converting categorical inputs into numerical format
    gender = 1 if gender == "Female" else 0
    senior_citizen = 1 if senior_citizen == "Yes" else 0
    partner = 1 if partner == "Yes" else 0
    dependents = 1 if dependents == "Yes" else 0
    phone_service = 1 if phone_service == "Yes" else 0
    multiple_lines = 1 if multiple_lines == "Yes" else 0
    internet_service = 2 if internet_service == "Fiber optic" else (1 if internet_service == "DSL" else 0)
    online_security = 1 if online_security == "Yes" else 0
    online_backup = 1 if online_backup == "Yes" else 0
    device_protection = 1 if device_protection == "Yes" else 0
    tech_support = 1 if tech_support == "Yes" else 0
    streaming_tv = 1 if streaming_tv == "Yes" else 0
    streaming_movies = 1 if streaming_movies == "Yes" else 0
    contract = 0 if contract == "Month-to-month" else (1 if contract == "One year" else 2)
    paperless_billing = 1 if paperless_billing == "Yes" else 0
    payment_method = {"Electronic check": 0, "Credit card (automatic)": 1, "Mailed check": 2, "Bank transfer (automatic)": 3}
    payment_method = payment_method[payment_method]
    
    # Prepare features for prediction
    features = np.array([[gender, senior_citizen, partner, dependents, phone_service, multiple_lines, 
                          internet_service, online_security, online_backup, device_protection, tech_support, 
                          streaming_tv, streaming_movies, contract, paperless_billing, payment_method, 
                          tenure, monthly_charges, total_charges]])
    
    # Display prediction button
    if st.button("Predict"):
        model = load_model(model_option)  # Load the selected model
        
        # Predict churn probability using the selected model
        churn_probability = model.predict_proba(features)[0][1] * 100  # Probability of churn
        prediction = "Stay" if churn_probability < 50 else "Churn"
        
        st.subheader("Prediction Results")
        st.write(f"The customer has a {churn_probability:.2f}% probability of staying.")
        st.write(f"Prediction: {prediction}")
        
        # Option to try another model
        if st.button("Do you want to try another model?"):
            st.experimental_rerun()  # Refresh the page to try a new model

# Main function to control the flow
def main():
    # Display the churn prediction page
    churn_page()

if __name__ == "__main__":
    main()
