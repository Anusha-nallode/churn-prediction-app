import pickle

with open("logistic_regression.pkl", "rb") as file:
    model = pickle.load(file)

print("Loaded Model Type:", type(model))
