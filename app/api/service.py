from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

MODEL_PATH = "artifacts/final_disaster_classifier.pkl"


class DisasterModel:
    """Wrapper around the trained ML model for disaster prediction."""

    def __init__(self, model_path: str = MODEL_PATH):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def predict_event(self, input_data: dict) -> int:
        """Run prediction on a single disaster event."""
        df = pd.DataFrame([input_data])

        # Ensure categorical features are strings
        categorical_features = ["disaster_type", "location", "aid_provided"]
        for col in categorical_features:
            if col in df.columns:
                df[col] = df[col].astype(str)

        prediction = self.model.predict(df)
        return int(prediction[0])


# FastAPI app
app = FastAPI(title="Disaster Analysis ML API")

# Load model once at startup
model = DisasterModel()


class DisasterEvent(BaseModel):
    disaster_type: str
    location: str
    latitude: float
    longitude: float
    severity_level: int
    affected_population: int
    estimated_economic_loss_usd: float
    response_time_hours: float
    aid_provided: int
    infrastructure_damage_index: float
    year: int
    month: int
    day: int
    day_of_week: int


@app.post("/predict")
def predict_disaster_event(event: DisasterEvent):
    """API endpoint to predict whether a disaster is major or minor."""
    result = model.predict_event(event.dict())
    return {"prediction": result}
