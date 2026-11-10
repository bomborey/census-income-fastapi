"""FastAPI app that serves the census income classification model."""
from pathlib import Path
from typing import Dict

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from starter.ml.data import process_data
from starter.ml.model import inference, load_artifact

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

model = load_artifact(MODEL_DIR / "model.joblib")
encoder = load_artifact(MODEL_DIR / "encoder.joblib")
lb = load_artifact(MODEL_DIR / "lb.joblib")


class CensusRecord(BaseModel):
    """One person's census record. Hyphenated names are handled by aliases."""

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "examples": [
                {
                    "age": 52,
                    "workclass": "Self-emp-inc",
                    "fnlgt": 287927,
                    "education": "HS-grad",
                    "education-num": 9,
                    "marital-status": "Married-civ-spouse",
                    "occupation": "Exec-managerial",
                    "relationship": "Wife",
                    "race": "White",
                    "sex": "Female",
                    "capital-gain": 15024,
                    "capital-loss": 0,
                    "hours-per-week": 40,
                    "native-country": "United-States",
                }
            ]
        },
    )

    age: int = Field(..., examples=[52])
    workclass: str = Field(..., examples=["Self-emp-inc"])
    fnlgt: int = Field(..., examples=[287927])
    education: str = Field(..., examples=["HS-grad"])
    education_num: int = Field(..., alias="education-num", examples=[9])
    marital_status: str = Field(
        ..., alias="marital-status", examples=["Married-civ-spouse"]
    )
    occupation: str = Field(..., examples=["Exec-managerial"])
    relationship: str = Field(..., examples=["Wife"])
    race: str = Field(..., examples=["White"])
    sex: str = Field(..., examples=["Female"])
    capital_gain: int = Field(..., alias="capital-gain", examples=[15024])
    capital_loss: int = Field(..., alias="capital-loss", examples=[0])
    hours_per_week: int = Field(..., alias="hours-per-week", examples=[40])
    native_country: str = Field(
        ..., alias="native-country", examples=["United-States"]
    )


app = FastAPI(
    title="Census Income Prediction API",
    description="Predicts whether a person's income is >50K or <=50K.",
    version="1.0.0",
)


@app.get("/")
async def welcome() -> Dict[str, str]:
    """Return a greeting."""
    return {"message": "Welcome to the Census Income Prediction API!"}


@app.post("/predict")
async def predict(record: CensusRecord) -> Dict[str, str]:
    """Run model inference on one census record."""
    df = pd.DataFrame([record.model_dump(by_alias=True)])
    X, _, _, _ = process_data(
        df,
        categorical_features=CAT_FEATURES,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    pred = inference(model, X)
    label = lb.inverse_transform(pred)[0]
    return {"prediction": str(label)}
