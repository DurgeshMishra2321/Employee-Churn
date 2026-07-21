# Employee Attrition Predictor

A binary classifier that predicts whether an employee is about to leave,
wrapped in a full pipeline from raw HR data to a live BI dashboard: data
prep → model training with experiment tracking → containerized REST API →
CI/CD → Power BI monitoring dashboard.

> "Built and deployed an employee attrition prediction system with a full
> MLOps pipeline — experiment tracking, containerized API serving, CI/CD,
> and a Power BI dashboard for live monitoring."

## Dataset

[IBM HR Analytics Employee Attrition](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
— 1,470 employees, 35 columns, pseudonymized IBM data. `data/employee_attrition.csv`.

Real-world messiness handled in [`src/data_prep.py`](src/data_prep.py):
- `EmployeeCount`, `StandardHours`, `Over18` are constant across every row — dropped as zero-signal.
- `EmployeeNumber` is a row ID — dropped to avoid leaking an arbitrary identifier into the model.
- Engineered features: `TenureBucket` (binned `YearsAtCompany`), `IncomePerJobLevel`,
  `PromotionGapRatio`, `ManagerTenureRatio`.

## Architecture

```
1. Load & clean data      ->  pandas               (src/data_prep.py)
2. Train the model        ->  scikit-learn          (src/train.py)
3. Track experiments      ->  MLflow                (mlruns/, mlflow.db)
4. Package & serve        ->  FastAPI + Docker       (src/serve.py, Dockerfile)
5. Automate the pipeline  ->  GitHub Actions         (.github/workflows/ci.yml)
6. Visualize & monitor    ->  Power BI               (reports/)
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. Train (logs every run to MLflow, saves the best model by recall)
python src/train.py

# 2. Cross-check the manual numpy metrics against sklearn
python src/evaluate.py

# 3. Generate the CSVs Power BI reads from
python src/export_reports.py

# 4. Run tests (data checks, metric cross-check, model quality gate, API contract)
pytest -v

# 5. Serve the model
uvicorn serve:app --app-dir src --reload
# -> Swagger UI at http://127.0.0.1:8000/docs, use the pre-filled example to try /predict

# 6. Inspect experiment runs
mlflow ui
```

## Why recall as the model-selection metric

Missing an employee who's about to leave (a false negative) costs more than
a false alarm (a false positive) — a false negative means no retention
conversation happens at all. `src/train.py` selects the best run by recall,
not accuracy, and the CI quality gate (`tests/test_model.py`) fails the
build if recall drops below 0.55.

## Model quality gate

`tests/test_model.py::test_trained_model_meets_quality_gate` requires
`recall >= 0.55` and `accuracy >= 0.70` on the held-out test set. CI trains
the model fresh on every push and runs this gate before building the Docker
image — a regression in model quality fails the build, not just the tests.

## API

`POST /predict` — takes a full employee record (see `/docs` for the schema
and a pre-filled example), returns `{"attrition_prediction": "Yes"|"No",
"attrition_probability": float}`. Every prediction is appended to
`reports/exports/prediction_log.csv` (input features + prediction + UTC
timestamp).

`GET /drift` — compares the mean of each numeric feature across all logged
predictions against the training-set mean, in standard-deviation units
(z-score). Flags any feature where `|z| > 2`. This is a deliberately simple
signal, not a formal drift test — see [Scope guardrails](#scope-guardrails).

`GET /health` — liveness check.

## Docker

```bash
docker build -t attrition-predictor .
docker run -p 8000:8000 attrition-predictor
```

The image only copies `src/` and `models/` — it serves a pre-trained model,
it doesn't train one. Build after running `train.py` (CI does this
automatically: it trains, tests, uploads the model as a build artifact, then
a second job downloads it and builds the image).

## CI/CD

`.github/workflows/ci.yml`, two jobs:
1. **lint-and-test** — ruff lint, train the model, run pytest (including the
   quality gate), upload `models/` as a build artifact.
2. **docker-build** — download that artifact, build the Docker image.

A model that fails the recall/accuracy gate never reaches the Docker build.

## Power BI dashboard

Three pages, fed by `reports/exports/*.csv`:

| Page | Source CSVs | Contents |
|---|---|---|
| **EDA** | `employee_data_clean.csv` | Attrition rate by department/overtime/job role/tenure bucket, slicers, a DAX measure for attrition rate (`DIVIDE(COUNTROWS(FILTER(...,Attrition="Yes")), COUNTROWS(...))`) |
| **Model performance** | `confusion_matrix.csv`, `model_metrics.csv`, `roc_curve.csv`, `feature_importance.csv` | Confusion matrix as a matrix visual, ROC curve (`fpr` x, `tpr` y, line chart), feature importance bar chart |
| **Live monitoring** | `prediction_log.csv` | Prediction volume over time (line chart on `timestamp`), predicted class balance (stacked bar), `/drift` z-scores pulled in manually as a KPI card |

Refresh manually before a demo (Get Data → Text/CSV → point at
`reports/exports/`) — no Power BI Gateway / scheduled cloud refresh needed
for a portfolio project.

## Repo structure

```
churn-predictor/
├── data/employee_attrition.csv
├── src/
│   ├── data_prep.py       # load, clean, feature-engineer
│   ├── train.py            # train + MLflow tracking, saves models/model.joblib
│   ├── evaluate.py         # manual numpy metrics, cross-checked vs sklearn
│   ├── serve.py             # FastAPI app: /predict, /drift, /health
│   └── export_reports.py   # writes reports/exports/*.csv for Power BI
├── tests/test_model.py
├── reports/exports/         # CSVs for Power BI (gitignored except structure)
├── Dockerfile
├── .github/workflows/ci.yml
└── requirements.txt
```

## Scope guardrails

Kept simple on purpose:
- No Kubernetes — a single container is enough.
- No real-time streaming — batch retraining, manual dashboard refresh.
- No formal drift statistics (PSI, KS-test) — `/drift` is a z-score
  comparison of live vs. training feature means. At scale, this would be
  replaced with PSI per feature, a rolling window instead of "all logged
  predictions," and alerting on sustained drift rather than a single flag.
- No Power BI Service/Pro/Gateway — Desktop + manual refresh (or Publish to
  Web) is enough for a portfolio.
- One dataset, one model — resisted adding a second use case.

## Interview talking points

- **Why logistic regression won over random forest** — despite lower raw
  accuracy, logistic regression had meaningfully higher recall (0.68 vs.
  0.17 in the reference run) on this class-imbalanced problem
  (~16% attrition rate). `class_weight="balanced"` matters a lot here.
- **Why recall over accuracy** — see [above](#why-recall-as-the-model-selection-metric).
- **How model iterations are managed** — MLflow experiment tracking, not
  manually renamed files; every run's params/metrics/model artifact are
  logged and comparable in `mlflow ui`.
- **How you'd know if this breaks in production** — the `/drift` signal,
  plus what a production version would add (PSI, alerting, scheduled
  retraining).
- **Deployment** — Docker container behind FastAPI, built and validated
  automatically by CI on every push, gated on a model-quality test.
- **Why Power BI instead of a Python dashboard** — it's the tool
  business/analytics teams actually use; shows the ability to bridge from
  "shipped a model" to "the report the business looks at."
