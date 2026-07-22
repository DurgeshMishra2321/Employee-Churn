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
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

from data_prep import ALL_FEATURES, NUMERIC_FEATURES
from recommend import generate_recommendations

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


def _engineer_dict(row: dict) -> pd.DataFrame:
    df = pd.DataFrame([row])
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


def _engineer(record: EmployeeRecord) -> pd.DataFrame:
    return _engineer_dict(record.model_dump())


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


_DEMO_HTML = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Employee Attrition Predictor - Dashboard</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body {
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    max-width: 1180px; margin: 0 auto; padding: 24px 20px 60px;
    background: #f8fafc; color: #0f172a;
  }
  h1 { font-size: 1.5rem; margin-bottom: 4px; }
  .subtitle { color: #64748b; margin-top: 0; margin-bottom: 20px; font-size: 0.95rem; }
  .toolbar { display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }
  button {
    font-size: 0.9rem; padding: 9px 16px; border-radius: 8px; border: none;
    cursor: pointer; color: white; font-weight: 600;
  }
  #lowBtn { background: #2563eb; }
  #highBtn { background: #dc2626; }
  #predictBtn { background: #16a34a; font-size: 1rem; padding: 12px 24px; }
  #predictBtn:disabled { background: #94a3b8; cursor: default; }

  .layout { display: grid; grid-template-columns: 1.15fr 1fr; gap: 20px; align-items: start; }
  @media (max-width: 900px) { .layout { grid-template-columns: 1fr; } }

  .card {
    background: white; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 18px 20px; margin-bottom: 16px;
  }
  .card h2 { font-size: 1rem; margin: 0 0 14px; color: #334155; }

  .field-group { margin-bottom: 16px; }
  .field-group-title {
    font-size: 0.75rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.04em; color: #94a3b8; margin-bottom: 8px;
  }
  .field-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px 14px; }
  .field label { display: block; font-size: 0.78rem; color: #475569; margin-bottom: 3px; }
  .field input, .field select {
    width: 100%; padding: 7px 8px; border-radius: 6px; border: 1px solid #cbd5e1;
    font-size: 0.85rem; background: white; color: #0f172a;
  }

  #placeholder { color: #94a3b8; text-align: center; padding: 40px 20px; }

  .badge {
    display: inline-block; padding: 6px 14px; border-radius: 999px;
    font-weight: 700; font-size: 0.95rem;
  }
  .badge.no { background: #dcfce7; color: #166534; }
  .badge.yes { background: #fee2e2; color: #991b1b; }

  .gauge-wrap { margin: 16px 0 4px; }
  .gauge-track { height: 22px; border-radius: 11px; background: #e2e8f0; overflow: hidden; }
  .gauge-fill { height: 100%; border-radius: 11px; transition: width 0.4s ease; }
  .gauge-label { display: flex; justify-content: space-between; font-size: 0.78rem; color: #64748b; margin-top: 4px; }
  .proba-big { font-size: 2.1rem; font-weight: 800; margin: 2px 0 0; }

  .rec-card {
    border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px; margin-bottom: 10px;
    background: #f8fafc;
  }
  .rec-card .rec-title { font-weight: 700; font-size: 0.92rem; margin-bottom: 3px; }
  .rec-card .rec-desc { font-size: 0.82rem; color: #475569; margin-bottom: 8px; }
  .rec-bars { display: flex; align-items: center; gap: 8px; font-size: 0.75rem; color: #64748b; }
  .rec-bar-track { flex: 1; height: 10px; border-radius: 5px; background: #e2e8f0; position: relative; overflow: hidden; }
  .rec-bar-fill { position: absolute; top: 0; left: 0; height: 100%; background: #16a34a; border-radius: 5px; }
  .rec-impact { font-weight: 700; color: #16a34a; min-width: 54px; text-align: right; }

  #noRecs { color: #64748b; font-size: 0.9rem; padding: 8px 0; }
  .footer-note { font-size: 0.75rem; color: #94a3b8; margin-top: 24px; }
</style>
</head>
<body>
<h1>Employee Attrition Dashboard</h1>
<p class="subtitle">Enter (or load) an employee record. The model scores attrition risk, and the recommendation engine re-runs the same model with each actionable change to measure what would actually reduce that employee's risk.</p>

<div class="toolbar">
  <button id="lowBtn">Load low-risk example</button>
  <button id="highBtn">Load high-risk example</button>
</div>

<div class="layout">
  <div class="card">
    <h2>Employee record</h2>
    <div id="formHost"></div>
    <div class="toolbar" style="margin-top: 4px;">
      <button id="predictBtn">Predict &amp; get recommendations</button>
    </div>
  </div>

  <div>
    <div class="card">
      <h2>Predicted risk</h2>
      <div id="riskPanel">
        <div id="placeholder">Fill in the form and click Predict.</div>
      </div>
    </div>

    <div class="card" id="recCard" style="display:none;">
      <h2>Suggested actions to reduce this employee's attrition risk</h2>
      <div id="recHost"></div>
    </div>
  </div>
</div>

<p class="footer-note">Recommendations are counterfactual simulations against the trained model (not a fixed rule list) — each action re-scores the employee with that one field changed, and only actions that measurably reduce risk are shown, ranked by impact.</p>

<script>
const DEPARTMENTS = ["Human Resources", "Research & Development", "Sales"];
const EDUCATION_FIELDS = ["Human Resources", "Life Sciences", "Marketing", "Medical", "Other", "Technical Degree"];
const JOB_ROLES = ["Healthcare Representative", "Human Resources", "Laboratory Technician", "Manager",
  "Manufacturing Director", "Research Director", "Research Scientist", "Sales Executive", "Sales Representative"];

const FIELDS = [
  { group: "Role & demographics", name: "Age", type: "number", min: 18, max: 70 },
  { group: "Role & demographics", name: "Gender", type: "select", options: ["Male", "Female"] },
  { group: "Role & demographics", name: "MaritalStatus", type: "select", options: ["Single", "Married", "Divorced"] },
  { group: "Role & demographics", name: "Education", type: "number", min: 1, max: 5 },
  { group: "Role & demographics", name: "EducationField", type: "select", options: EDUCATION_FIELDS },
  { group: "Role & demographics", name: "Department", type: "select", options: DEPARTMENTS },
  { group: "Role & demographics", name: "JobRole", type: "select", options: JOB_ROLES },
  { group: "Role & demographics", name: "JobLevel", type: "number", min: 1, max: 5 },

  { group: "Compensation", name: "MonthlyIncome", type: "number", min: 1000 },
  { group: "Compensation", name: "DailyRate", type: "number", min: 0 },
  { group: "Compensation", name: "HourlyRate", type: "number", min: 0 },
  { group: "Compensation", name: "MonthlyRate", type: "number", min: 0 },
  { group: "Compensation", name: "PercentSalaryHike", type: "number", min: 0 },
  { group: "Compensation", name: "StockOptionLevel", type: "number", min: 0, max: 3 },
  { group: "Compensation", name: "PerformanceRating", type: "number", min: 1, max: 4 },

  { group: "Work conditions", name: "OverTime", type: "select", options: ["Yes", "No"] },
  { group: "Work conditions", name: "BusinessTravel", type: "select", options: ["Non-Travel", "Travel_Rarely", "Travel_Frequently"] },
  { group: "Work conditions", name: "DistanceFromHome", type: "number", min: 0 },
  { group: "Work conditions", name: "WorkLifeBalance", type: "number", min: 1, max: 4 },
  { group: "Work conditions", name: "EnvironmentSatisfaction", type: "number", min: 1, max: 4 },
  { group: "Work conditions", name: "JobSatisfaction", type: "number", min: 1, max: 4 },
  { group: "Work conditions", name: "JobInvolvement", type: "number", min: 1, max: 4 },
  { group: "Work conditions", name: "RelationshipSatisfaction", type: "number", min: 1, max: 4 },
  { group: "Work conditions", name: "TrainingTimesLastYear", type: "number", min: 0 },

  { group: "Tenure & history", name: "NumCompaniesWorked", type: "number", min: 0 },
  { group: "Tenure & history", name: "TotalWorkingYears", type: "number", min: 0 },
  { group: "Tenure & history", name: "YearsAtCompany", type: "number", min: 0 },
  { group: "Tenure & history", name: "YearsInCurrentRole", type: "number", min: 0 },
  { group: "Tenure & history", name: "YearsSinceLastPromotion", type: "number", min: 0 },
  { group: "Tenure & history", name: "YearsWithCurrManager", type: "number", min: 0 },
];

const lowRisk = {
  "Age": 45, "DailyRate": 1200, "DistanceFromHome": 2, "Education": 4,
  "EnvironmentSatisfaction": 4, "HourlyRate": 90, "JobInvolvement": 4, "JobLevel": 4,
  "JobSatisfaction": 4, "MonthlyIncome": 14000, "MonthlyRate": 20000, "NumCompaniesWorked": 1,
  "PercentSalaryHike": 18, "PerformanceRating": 4, "RelationshipSatisfaction": 4,
  "StockOptionLevel": 2, "TotalWorkingYears": 20, "TrainingTimesLastYear": 3,
  "WorkLifeBalance": 4, "YearsAtCompany": 15, "YearsInCurrentRole": 10,
  "YearsSinceLastPromotion": 1, "YearsWithCurrManager": 8, "BusinessTravel": "Non-Travel",
  "Department": "Research & Development", "EducationField": "Medical", "Gender": "Male",
  "JobRole": "Manager", "MaritalStatus": "Married", "OverTime": "No"
};
const highRisk = {
  "Age": 24, "DailyRate": 300, "DistanceFromHome": 25, "Education": 2,
  "EnvironmentSatisfaction": 1, "HourlyRate": 40, "JobInvolvement": 1, "JobLevel": 1,
  "JobSatisfaction": 1, "MonthlyIncome": 2200, "MonthlyRate": 8000, "NumCompaniesWorked": 4,
  "PercentSalaryHike": 11, "PerformanceRating": 3, "RelationshipSatisfaction": 1,
  "StockOptionLevel": 0, "TotalWorkingYears": 2, "TrainingTimesLastYear": 0,
  "WorkLifeBalance": 1, "YearsAtCompany": 1, "YearsInCurrentRole": 0,
  "YearsSinceLastPromotion": 0, "YearsWithCurrManager": 0, "BusinessTravel": "Travel_Frequently",
  "Department": "Sales", "EducationField": "Marketing", "Gender": "Male",
  "JobRole": "Sales Representative", "MaritalStatus": "Single", "OverTime": "Yes"
};

function buildForm() {
  const host = document.getElementById("formHost");
  const groups = [...new Set(FIELDS.map(f => f.group))];
  host.innerHTML = groups.map(g => {
    const fields = FIELDS.filter(f => f.group === g);
    const inputs = fields.map(f => {
      const id = "f_" + f.name;
      if (f.type === "select") {
        const opts = f.options.map(o => `<option value="${o}">${o}</option>`).join("");
        return `<div class="field"><label for="${id}">${f.name}</label><select id="${id}">${opts}</select></div>`;
      }
      const minAttr = f.min !== undefined ? ` min="${f.min}"` : "";
      const maxAttr = f.max !== undefined ? ` max="${f.max}"` : "";
      return `<div class="field"><label for="${id}">${f.name}</label><input type="number" id="${id}"${minAttr}${maxAttr}></div>`;
    }).join("");
    return `<div class="field-group"><div class="field-group-title">${g}</div><div class="field-grid">${inputs}</div></div>`;
  }).join("");
}

function loadValues(values) {
  FIELDS.forEach(f => {
    const el = document.getElementById("f_" + f.name);
    if (el && values[f.name] !== undefined) el.value = values[f.name];
  });
}

function readValues() {
  const values = {};
  FIELDS.forEach(f => {
    const el = document.getElementById("f_" + f.name);
    values[f.name] = f.type === "number" ? Number(el.value) : el.value;
  });
  return values;
}

function riskColor(p) {
  if (p < 0.3) return "#16a34a";
  if (p < 0.6) return "#f59e0b";
  return "#dc2626";
}

function renderRisk(prediction, probability) {
  const pct = (probability * 100).toFixed(1);
  const color = riskColor(probability);
  document.getElementById("riskPanel").innerHTML = `
    <span class="badge ${prediction === 'Yes' ? 'yes' : 'no'}">${prediction === 'Yes' ? 'Likely to leave' : 'Likely to stay'}</span>
    <div class="proba-big" style="color:${color}">${pct}%</div>
    <div class="gauge-wrap">
      <div class="gauge-track"><div class="gauge-fill" style="width:${pct}%; background:${color};"></div></div>
      <div class="gauge-label"><span>0%</span><span>Attrition probability</span><span>100%</span></div>
    </div>
  `;
}

function renderRecommendations(recs) {
  const card = document.getElementById("recCard");
  const host = document.getElementById("recHost");
  card.style.display = "block";

  if (!recs.length) {
    host.innerHTML = `<div id="noRecs">No actionable change moved this employee's predicted risk down — this profile is already low-risk given the model.</div>`;
    return;
  }

  const maxImpact = Math.max(...recs.map(r => r.probability_reduction));
  host.innerHTML = recs.map(r => {
    const widthPct = Math.max(6, (r.probability_reduction / maxImpact) * 100);
    return `
      <div class="rec-card">
        <div class="rec-title">${r.label}</div>
        <div class="rec-desc">${r.description}</div>
        <div class="rec-bars">
          <span>${(r.baseline_probability * 100).toFixed(1)}% &rarr; ${(r.new_probability * 100).toFixed(1)}%</span>
          <div class="rec-bar-track"><div class="rec-bar-fill" style="width:${widthPct}%;"></div></div>
          <span class="rec-impact">-${(r.probability_reduction * 100).toFixed(1)} pts</span>
        </div>
      </div>
    `;
  }).join("");
}

async function predict() {
  const btn = document.getElementById("predictBtn");
  btn.disabled = true;
  btn.textContent = "Predicting...";
  document.getElementById("riskPanel").innerHTML = `<div id="placeholder">Scoring employee and simulating interventions...</div>`;
  try {
    const payload = readValues();
    const resp = await fetch("/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await resp.json();
    if (!resp.ok) {
      document.getElementById("riskPanel").innerHTML = `<div id="placeholder">Error: ${JSON.stringify(data)}</div>`;
      return;
    }
    renderRisk(data.baseline_prediction, data.baseline_probability);
    renderRecommendations(data.recommendations);
  } catch (e) {
    document.getElementById("riskPanel").innerHTML = `<div id="placeholder">Request failed: ${e}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Predict & get recommendations";
  }
}

buildForm();
loadValues(lowRisk);
document.getElementById("lowBtn").onclick = () => loadValues(lowRisk);
document.getElementById("highBtn").onclick = () => loadValues(highRisk);
document.getElementById("predictBtn").onclick = predict;
</script>
</body>
</html>
"""


@app.get("/demo", response_class=HTMLResponse)
def demo():
    return _DEMO_HTML


@app.post("/predict", response_model=PredictionResponse)
def predict(record: EmployeeRecord):
    model = get_model()
    features = _engineer(record)

    proba = float(model.predict_proba(features)[0, 1])
    prediction = "Yes" if proba >= 0.5 else "No"

    _log_prediction(features, prediction, proba)

    return PredictionResponse(attrition_prediction=prediction, attrition_probability=proba)


@app.post("/recommend")
def recommend(record: EmployeeRecord):
    """Predict attrition risk for this employee, then simulate company-actionable
    changes (overtime, satisfaction, pay, promotion, ...) through the same model
    to see which ones actually move the predicted probability, ranked by impact."""
    model = get_model()
    base_row = record.model_dump()

    features = _engineer_dict(base_row)
    proba = float(model.predict_proba(features)[0, 1])
    prediction = "Yes" if proba >= 0.5 else "No"
    _log_prediction(features, prediction, proba)

    result = generate_recommendations(model, _engineer_dict, base_row, top_n=5)
    return result


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
