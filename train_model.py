import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 🎯 Generate Sample Training Data (Replace with real data)
np.random.seed(42)
X = np.random.rand(1000, 18)  # 1000 samples, 18 features
y = np.random.randint(0, 2, 1000)  # Binary target variable (0 or 1)

# Split into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train Logistic Regression
logistic_model = LogisticRegression(max_iter=500)
logistic_model.fit(X_train, y_train)

# ✅ Train Random Forest
random_forest_model = RandomForestClassifier(n_estimators=100)
random_forest_model.fit(X_train, y_train)

# Save Logistic Regression Model
with open("logistic_regression.pkl", "wb") as f:
    pickle.dump(logistic_model, f)

# Save Random Forest Model
with open("random_forest.pkl", "wb") as f:
    pickle.dump(random_forest_model, f)

print("✅ Models trained and saved successfully!")
