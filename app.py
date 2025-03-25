import streamlit as st
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd  # ✅ Add this line to fix the error


# ✅ Set page config at the very top
st.set_page_config(page_title="Churn Prediction", layout="wide")

# ✅ Initialize session state at the very beginning
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "history" not in st.session_state:
    st.session_state["history"] = []

# Load models with error handling
def load_model(model_name):
    if not os.path.exists(model_name):
        st.error(f"⚠️ Error: Model file `{model_name}` not found. Please train and save the model first.")
        return None
    try:
        with open(model_name, "rb") as f:
            model = pickle.load(f)
        return model
    except pickle.UnpicklingError:
        st.error("❌ Error: Failed to load the model. The file might be corrupted or saved incorrectly.")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error while loading model: {str(e)}")
        return None

# Initialize session state for login
def login():
    if st.session_state["logged_in"]:
        return

    st.markdown("<h2 style='text-align: center;'>🔒 Login</h2>", unsafe_allow_html=True)

    username = st.text_input("Username", placeholder="Enter your username", key="unique_login_user")
    password = st.text_input("Password", placeholder="Enter your password", type="password", key="unique_login_pass")

    if st.button("Login", key="unique_login_btn"):
        if username == "user" and password == "abc123":
            st.session_state["logged_in"] = True
            st.rerun()
        else:
            st.error("❌ Invalid credentials. Try again.")

    st.info("🔑 Please log in to continue.")
    st.markdown("### Demo Credentials")
    st.markdown("**Username:** `user`  \n**Password:** `abc123`")

if not st.session_state["logged_in"]:
    login()
    st.stop()

def main():
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
        st.title("📊 Churn Predictor")
        page = st.radio("", ["Home", "Predict", "History", "Dashboard"], key="unique_nav")

        if st.button("🚪 Logout", key="unique_logout_btn"):
            st.session_state["logged_in"] = False
            st.rerun()

    if page == "Home":
        st.markdown("<h1 style='text-align: center;'>📈 Welcome to Churn Predictor</h1>", unsafe_allow_html=True)
        st.markdown("""
        ### Why Does Bank Customer Churn Happen? 🏦
        - **High Fees & Charges** 💰
        - **Poor Customer Service** ☎️
        - **Limited Digital Banking Features** 📱
        - **Better Offers from Competitors** 🎯
        - **Security Concerns** 🔐
        
        **🔍 How Can AI Help?**
        AI models can analyze customer behavior and predict churn **before it happens**, allowing banks to take preventive actions and retain valuable customers.
        """, unsafe_allow_html=True)

    elif page == "Predict":
        st.markdown("<h2 style='text-align: center;'> Select a Model and Predict the Future 🔮</h2>", unsafe_allow_html=True)
        model_choice = st.selectbox("Select Prediction Model", ["Logistic Regression", "Random Forest"], key="unique_model_select")

        model_filename = "logistic_regression.pkl" if model_choice == "Logistic Regression" else "random_forest.pkl"
        model = load_model(model_filename)
        if model is None:
            return

        st.markdown("<h2 style='text-align: center;'>📌 Select Features for Prediction 🤖 </h2>", unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            gender = st.radio("Gender", ["Male", "Female"], key="gender")
            senior_citizen = st.radio("Senior Citizen", [0, 1], key="senior")
            partner = st.radio("Partner", ["Yes", "No"], key="partner")
            dependents = st.radio("Dependents", ["Yes", "No"], key="dependents")

        with col2:
            phone_service = st.radio("Phone Service", ["Yes", "No"], key="phone_service")
            multiple_lines = st.radio("Multiple Lines", ["Yes", "No"], key="multiple_lines")
            internet_service = st.radio("Internet Service", ["Fiber optic", "DSL", "No"], key="internet_service")
            online_security = st.radio("Online Security", ["Yes", "No"], key="online_security")

        with col3:
            online_backup = st.radio("Online Backup", ["Yes", "No"], key="online_backup")
            device_protection = st.radio("Device Protection", ["Yes", "No"], key="device_protection")
            tech_support = st.radio("Tech Support", ["Yes", "No"], key="tech_support")
            streaming_tv = st.radio("Streaming TV", ["Yes", "No"], key="streaming_tv")

        with col4:
            contract = st.radio("Contract", ["Month-to-month", "Two year", "One year"], key="contract")
            paperless_billing = st.radio("Paperless Billing", ["Yes", "No"], key="paperless_billing")
            payment_method = st.radio("Payment Method", ["Electronic check", "Credit card (automatic)", "Mailed check", "Bank transfer (automatic)"], key="payment_method")

        tenure = st.slider("Tenure (months)", min_value=1, max_value=72, value=30, key="tenure")
        monthly_charges = st.slider("Monthly Charges ($)", min_value=1, max_value=100, value=50, key="monthly_charges")
        total_charges = tenure * monthly_charges

        st.markdown(f"<h3 style='text-align: center;'>Total Charges: <span style='color: #FF4B4B;'>{total_charges}</span></h3>", unsafe_allow_html=True)

        if st.button("🔍 Predict", key="predict_button"):
            input_data = np.array([[1 if gender == "Male" else 0, senior_citizen, 1 if partner == "Yes" else 0, 1 if dependents == "Yes" else 0,
                                    1 if phone_service == "Yes" else 0, 1 if multiple_lines == "Yes" else 0,
                                    {"Fiber optic": 2, "DSL": 1, "No": 0}[internet_service],
                                    1 if online_security == "Yes" else 0, 1 if online_backup == "Yes" else 0,
                                    1 if device_protection == "Yes" else 0, 1 if tech_support == "Yes" else 0,
                                    1 if streaming_tv == "Yes" else 0, {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract],
                                    1 if paperless_billing == "Yes" else 0, {"Electronic check": 0, "Credit card (automatic)": 1, "Mailed check": 2, "Bank transfer (automatic)": 3}[payment_method],
                                    tenure, monthly_charges, total_charges]]).reshape(1, -1)
            prediction = model.predict_proba(input_data)[0][1] * 100
            st.success(f"🚀 The customer has a **{prediction:.2f}% probability of churning**." if prediction > 50 else f"✅ The customer has a **{prediction:.2f}% probability of staying**.")

            # ✅ Limit history to the last 10 entries
            if len(st.session_state["history"]) > 10:
                st.session_state["history"].pop(0)

            st.session_state["history"].append({
                "Model": model_choice,
                "Monthly Charges": monthly_charges,
                "Partner": partner,
                "Senior Citizen": senior_citizen,
                "Multiple Lines": multiple_lines,
                "Device Protection": device_protection,
                "Paperless Billing": paperless_billing,
                "Internet Service": internet_service,
                "Tenure": tenure,
                "Total Charges": total_charges,
                "Prediction": f"{prediction:.2f}%"
            })

    elif page == "History":
        st.markdown("<h2 style='text-align: center;'>📜 Prediction History</h2>", unsafe_allow_html=True)
        if not st.session_state["history"]:
            st.info("No prediction history available.")
        else:
            st.table(st.session_state["history"])

            if st.button("🗑️ Clear History"):
                st.session_state["history"] = []
                st.success("✅ History cleared!")

    elif page == "Dashboard":
        st.markdown("<h2 style='text-align: center;'>📊 Churn Analysis Dashboard</h2>", unsafe_allow_html=True)

        # Generate sample data
        if len(st.session_state["history"]) == 0:
            st.warning("No predictions available yet. Make some predictions first.")
            return

        df = pd.DataFrame(st.session_state["history"])

        # 📊 Churn Rate Pie Chart
        st.subheader("📊 Churn Rate Analysis")
        churned = sum(df["Prediction"].astype(str).str[:-1].astype(float) > 50)
        retained = len(df) - churned
        fig, ax = plt.subplots()
        ax.pie([churned, retained], labels=["Churned", "Retained"], autopct="%1.1f%%", colors=["red", "blue"])
        st.pyplot(fig)

        # 📈 Prediction Trend Line Chart
        st.subheader("📈 Prediction Trend Over Time")
        df["Index"] = range(1, len(df) + 1)
        fig, ax = plt.subplots()
        ax.plot(df["Index"], df["Prediction"].astype(str).str[:-1].astype(float), marker="o", linestyle="-")
        ax.set_xlabel("Prediction Number")
        ax.set_ylabel("Churn Probability (%)")
        ax.set_title("Churn Predictions Over Time")
        st.pyplot(fig)

        # 🔍 Feature Impact Bar Chart (Simulated)
        st.subheader("🔍 Feature Importance")
        feature_importance = {"Tenure": 30, "Monthly Charges": 50, "Total Charges": 20}
        fig, ax = plt.subplots()
        ax.bar(feature_importance.keys(), feature_importance.values(), color=["blue", "orange", "purple"])
        ax.set_ylabel("Importance (%)")
        st.pyplot(fig)

if __name__ == "__main__":
    main()
