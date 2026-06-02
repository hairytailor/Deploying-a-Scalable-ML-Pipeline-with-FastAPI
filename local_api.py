import json
import requests

# -----------------------------
# GET request to root endpoint
# -----------------------------
r = requests.get("http://127.0.0.1:8000/")
print("GET / status:", r.status_code)
print("GET / response:", r.json())


# -----------------------------
# Data for POST request
# -----------------------------
data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# -----------------------------
# POST request to /predict
# -----------------------------
r = requests.post("http://127.0.0.1:8000/predict", json=data)
print("POST /predict status:", r.status_code)
print("POST /predict response:", r.json())
