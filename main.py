from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel, Field
import joblib
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

COLUMNS = [
    "latitude",
    "longitude",
    "price",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
    "neighbourhood_group",
    "neighbourhood"
]

# Load the trained model pipeline
Model = joblib.load("Model_Pipeline.pkl")

# Input validation model
class Features(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, description="Latitude coordinate")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude coordinate")
    price: float = Field(..., gt=0, description="Price per night")
    minimum_nights: int = Field(..., ge=1, le=365, description="Minimum nights required")
    number_of_reviews: int = Field(..., ge=0, description="Total number of reviews")
    reviews_per_month: float = Field(..., ge=0, description="Average reviews per month")
    calculated_host_listings_count: int = Field(..., ge=0, description="Number of listings by this host")
    availability_365: int = Field(..., ge=0, le=365, description="Days available out of 365")
    neighbourhood_group: str = Field(..., min_length=1, description="Neighbourhood group")
    neighbourhood: str = Field(..., min_length=1, description="Neighbourhood name")

@app.get("/")
def greet():
    return {"message": "Hello Guys"}

@app.post("/predict")
def predict(features: Features):
    row = pd.DataFrame([features.model_dump()])[COLUMNS]

    prediction = Model.predict(row)
    probability = Model.predict_proba(row)

    return {
        "predicted_room": prediction[0],
        "probability": probability.tolist()[0]
    }