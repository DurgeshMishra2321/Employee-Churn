"""FastAPI serving layer for the employee attrition model.

- POST /predict: run inference on a single employee record
- GET  /health: liveness check
- GET  /drift: numpy-based comparison of recent prediction inputs vs the
  training feature distribution (a lightweight monitoring signal, not a
  formal drift test like PSI/KS — see README for what a production version
  would add)
"""

import csv
import os
from datetime import datetime, timezone
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from data_prep import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.joblib")
TRAINING_STATS_PATH = os.environ.get("TRAINING_STATS_PATH", "models/training_feature_stats.csv")
PREDICTION_LOG_PATH = os.environ.get("PREDICTION_LOG_PATH", "reports/exports/prediction_log.csv")

app = FastAPI(title="Employee Attrition Predictor", version="1.0.0")

_model = None
_training_stats = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(status_code=503, detail=f"Model not found at {MODEL_PATH}. Run train.py first.")
        _model = joblib.load(MODEL_PATH)
    return _model


def get_training_stats() -> pd.DataFrame:
    global _training_stats
    if _training_stats is None:
        if not os.path.exists(TRAINING_STATS_PATH):
            raise HTTPException(status_code=503, detail="Training feature stats not found. Run train.py first.")
        _training_stats = pd.read_csv(TRAINING_STATS_PATH, index_col=0)
    return _training_stats


class EmployeeRecord(BaseModel):
    Age: int = Field(..., ge=18, le=70)
    DailyRate: int
    DistanceFromHome: int = Field(..., ge=0)
    Education: int = Field(..., ge=1, le=5)
    EnvironmentSatisfaction: int = Field(..., ge=1, le=4)
    HourlyRate: int
    JobInvolvement: int = Field(..., ge=1, le=4)
    JobLevel: int = Field(..., ge=1, le=5)
    JobSatisfaction: int = Field(..., ge=1, le=4)
    MonthlyIncome: int = Field(..., gt=0)
    MonthlyRate: int
    NumCompaniesWorked: int = Field(..., ge=0)
    PercentSalaryHike: int
    PerformanceRating: int = Field(..., ge=1, le=4)
    RelationshipSatisfaction: int = Field(..., ge=1, le=4)
    StockOptionLevel: int = Field(..., ge=0, le=3)
    TotalWorkingYears: int = Field(..., ge=0)
    TrainingTimesLastYear: int = Field(..., ge=0)
    WorkLifeBalance: int = Field(..., ge=1, le=4)
    YearsAtCompany: int = Field(..., ge=0)
    YearsInCurrentRole: int = Field(..., ge=0)
    YearsSinceLastPromotion: int = Field(..., ge=0)
    YearsWithCurrManager: int = Field(..., ge=0)
    BusinessTravel: Literal["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
    Department: str
    EducationField: str
    Gender: Literal["Male", "Female"]
    JobRole: str
    MaritalStatus: Literal["Single", "Married", "Divorced"]
    OverTime: Literal["Yes", "No"]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Age": 34,
                "DailyRate": 800,
                "DistanceFromHome": 9,
                "Education": 3,
                "EnvironmentSatisfaction": 2,
                "HourlyRate": 65,
                "JobInvolvement": 3,
                "JobLevel": 2,
                "JobSatisfaction": 3,
                "MonthlyIncome": 5500,
                "MonthlyRate": 15000,
                "NumCompaniesWorked": 2,
                "PercentSalaryHike": 13,
                "PerformanceRating": 3,
                "RelationshipSatisfaction": 3,
                "StockOptionLevel": 1,
                "TotalWorkingYears": 8,
                "TrainingTimesLastYear": 2,
                "WorkLifeBalance": 3,
                "YearsAtCompany": 5,
                "YearsInCurrentRole": 3,
                "YearsSinceLastPromotion": 1,
                "YearsWithCurrManager": 2,
                "BusinessTravel": "Travel_Rarely",
                "Department": "Research & Development",
                "EducationField": "Life Sciences",
                "Gender": "Female",
                "JobRole": "Research Scientist",
                "MaritalStatus": "Married",
                "OverTime": "No",
            }
        }
    )


class PredictionResponse(BaseModel):
    attrition_prediction: Literal["Yes", "No"]
    attrition_probability: float


def _engineer(record: EmployeeRecord) -> pd.DataFrame:
    df = pd.DataFrame([record.model_dump()])
    df["TenureBucket"] = pd.cut(
        df["YearsAtCompany"],
        bins=[-np.inf, 1, 3, 7, 15, np.inf],
        labels=["<1yr", "1-3yr", "3-7yr", "7-15yr", "15yr+"],
    ).astype(str)
    df["IncomePerJobLevel"] = df["MonthlyIncome"] / df["JobLevel"].replace(0, 1)
    df["PromotionGapRatio"] = np.where(
        df["YearsAtCompany"] > 0, df["YearsSinceLastPromotion"] / df["YearsAtCompany"], 0.0
    )
    df["ManagerTenureRatio"] = np.where(
        df["YearsAtCompany"] > 0, df["YearsWithCurrManager"] / df["YearsAtCompany"], 0.0
    )
    return df[ALL_FEATURES]


def _log_prediction(features: pd.DataFrame, prediction: str, probability: float) -> None:
    os.makedirs(os.path.dirname(PREDICTION_LOG_PATH), exist_ok=True)
    row = features.iloc[0].to_dict()
    row["prediction"] = prediction
    row["probability"] = probability
    row["timestamp"] = datetime.now(timezone.utc).isoformat()

    file_exists = os.path.exists(PREDICTION_LOG_PATH)
    with open(PREDICTION_LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": os.path.exists(MODEL_PATH)}


@app.post("/predict", response_model=PredictionResponse)
def predict(record: EmployeeRecord):
    model = get_model()
    features = _engineer(record)

    proba = float(model.predict_proba(features)[0, 1])
    prediction = "Yes" if proba >= 0.5 else "No"

    _log_prediction(features, prediction, proba)

    return PredictionResponse(attrition_prediction=prediction, attrition_probability=proba)


@app.get("/drift")
def drift():
    """Compare the mean of each numeric feature in recent logged predictions
    against the training set mean, in standard-deviation units. This is a
    simple z-score style signal, not a formal drift test (PSI/KS) — flagged
    as a documented simplification, see README."""
    if not os.path.exists(PREDICTION_LOG_PATH):
        return {"status": "no_predictions_logged_yet"}

    stats = get_training_stats()
    log_df = pd.read_csv(PREDICTION_LOG_PATH)

    signals = {}
    for feature in NUMERIC_FEATURES:
        if feature not in log_df.columns or feature not in stats.columns:
            continue
        train_mean = stats.loc["mean", feature]
        train_std = stats.loc["std", feature] or 1.0
        live_mean = log_df[feature].astype(float).mean()
        z = (live_mean - train_mean) / train_std if train_std else 0.0
        signals[feature] = {
            "train_mean": round(float(train_mean), 3),
            "live_mean": round(float(live_mean), 3),
            "z_score": round(float(z), 3),
            "flagged": bool(abs(z) > 2),
        }

    return {
        "n_predictions": len(log_df),
        "signals": signals,
        "any_flagged": any(s["flagged"] for s in signals.values()),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
