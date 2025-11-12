import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Configuration constants
MODEL_PATH = os.path.join("models", "logistic_regression_iris.joblib")


# Input schema definition
class FlowerMeasurements(BaseModel):
    """Schema for iris flower measurements."""

    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# Initialize FastAPI application
application = FastAPI(
    title="Iris Species Predictor API",
    description="An API to predict the species of an Iris flower based on its features.",
    version="1.0.0",
)

# Load the trained model at startup
try:
    classifier_model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError(
        f"Model file not found at {MODEL_PATH}. Please run train.py first."
    )


@application.get("/", tags=["General"])
def health_check():
    """Health check endpoint."""
    return {"message": "Welcome to the Iris Species Predictor API!"}


@application.post("/predict", tags=["Prediction"])
def classify_iris(features: FlowerMeasurements):
    """
    Classifies the Iris species based on flower measurements.
    """
    # Transform input to DataFrame format expected by model
    feature_df = pd.DataFrame([features.dict()])

    # Generate prediction
    species_prediction = classifier_model.predict(feature_df)

    # Return result
    return {"predicted_species": species_prediction[0]}
