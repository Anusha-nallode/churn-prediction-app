import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
file_path = "C:\\Users\\Dell\\Downloads\\Customer_Churn.csv"
df = pd.read_csv(file_path)

# Drop 'customerID' as it's irrelevant
df.drop(columns=['customerID'], inplace=True)

# Convert 'TotalCharges' to numeric (handling missing values)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Convert 'Churn' column to binary (Yes → 1, No → 0)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Identify categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Encode categorical variables using one-hot encoding
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Define features (X) and target (y)
X = df.drop(columns=['Churn'])  # Features
y = df['Churn']  # Target variable

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing features (needed for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression Model
logistic_model = LogisticRegression()
logistic_model.fit(X_train_scaled, y_train)

# Train Random Forest Model
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)  # No scaling needed for Random Forest

# Save models and scaler
joblib.dump(logistic_model, 'logistic_regression.pkl')
joblib.dump(random_forest_model, 'random_forest.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("✅ Models trained and saved successfully.")

# Evaluate Models
logistic_pred = logistic_model.predict(X_test_scaled)
rf_pred = random_forest_model.predict(X_test)

print(f"🔹 Logistic Regression Accuracy: {accuracy_score(y_test, logistic_pred):.2f}")
print(f"🔹 Random Forest Accuracy: {accuracy_score(y_test, rf_pred):.2f}")

# Function to Load Models and Predict Churn
def predict_churn(model_type, new_data):
    """Predict churn using the specified model ('logistic_regression' or 'random_forest')"""
    
    if model_type == "logistic_regression":
        model = joblib.load('logistic_regression.pkl')
        scaler = joblib.load('scaler.pkl')
        new_data_scaled = scaler.transform([new_data])  # Scale input data
        prediction = model.predict(new_data_scaled)
    
    elif model_type == "random_forest":
        model = joblib.load('random_forest.pkl')
        prediction = model.predict([new_data])  # No scaling needed
    
    else:
        raise ValueError("❌ Invalid model type. Choose 'logistic_regression' or 'random_forest'.")
    
    return "Churn" if prediction[0] == 1 else "No Churn"

# Example input (replace with actual feature values from the dataset)
new_customer = X.iloc[0].values  # Using first row as an example input

# Predict churn for the new customer
print("🔹 Logistic Regression Prediction:", predict_churn("logistic_regression", new_customer))
print("🔹 Random Forest Prediction:", predict_churn("random_forest", new_customer))
