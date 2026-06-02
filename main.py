import os
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ml.data import process_data
from ml.model import inference, load_model


# -----------------------------
# Pydantic Data Model
# -----------------------------
class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(..., example="Married-civ-spouse", alias="marital-status")
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")

    class Config:
        allow_population_by_field_name = True


# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(
    title="Census Income Prediction API",
    description="Predict whether income >50K using a trained ML model.",
    version="1.0.0",
)


# -----------------------------
# Load model + encoder + lb on startup
# -----------------------------
@app.on_event("startup")
async def startup_event():
    global model, encoder, lb

    model = load_model("model/model.pkl")
    encoder = load_model("model/encoder.pkl")
    lb = load_model("model/lb.pkl") if os.path.exists("model/lb.pkl") else None


# -----------------------------
# GET Root Endpoint
# -----------------------------
@app.get("/")
async def root():
    return {"message": "Welcome to the Census Income Prediction API!"}


# -----------------------------
# POST Prediction Endpoint
# -----------------------------
@app.post("/predict")
async def predict(data: Data):
    # Convert Pydantic model → dict
    data_dict = data.dict(by_alias=True)

    # Convert to DataFrame
    df = pd.DataFrame([data_dict])

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    # Process input
    X, _, _, _ = process_data(
        df,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # Predict
    pred = inference(model, X)[0]

    # Convert numeric prediction → label
    label = ">50K" if pred == 1 else "<=50K"

    return {"prediction": label}
