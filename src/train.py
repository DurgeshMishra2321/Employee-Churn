"""Train baseline (logistic regression) and comparison (random forest) models
for employee attrition, tracking every run with MLflow and registering the
best one by recall (missing a departing employee costs more than a false
alarm on this problem)."""

import os

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_prep import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_dataset

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
MLFLOW_EXPERIMENT = "employee-attrition"
RANDOM_STATE = 42


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def build_candidates():
    """Return {name: (estimator, param_dict_for_logging)} for every model we compare."""
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
            {"model_type": "logistic_regression", "class_weight": "balanced", "max_iter": 1000},
        ),
        "random_forest": (
            RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            {
                "model_type": "random_forest",
                "n_estimators": 300,
                "max_depth": 8,
                "class_weight": "balanced",
            },
        ),
    }


def evaluate(pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def run_training():
    X, y = build_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    best_name, best_pipeline, best_metrics = None, None, None
    best_recall = -1.0

    for name, (estimator, params) in build_candidates().items():
        with mlflow.start_run(run_name=name):
            pipeline = Pipeline(
                steps=[("preprocess", build_preprocessor()), ("model", estimator)]
            )

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            cv_recall = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="recall")

            pipeline.fit(X_train, y_train)
            metrics = evaluate(pipeline, X_test, y_test)
            metrics["cv_recall_mean"] = float(np.mean(cv_recall))
            metrics["cv_recall_std"] = float(np.std(cv_recall))

            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipeline, name="model")

            print(f"[{name}] " + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))

            if metrics["recall"] > best_recall:
                best_recall = metrics["recall"]
                best_name, best_pipeline, best_metrics = name, pipeline, metrics

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"\nBest model: {best_name} (recall={best_recall:.3f}) -> saved to {MODEL_PATH}")

    # Persist training feature distributions for the drift check in serve.py.
    X_train.describe().to_csv(os.path.join(MODEL_DIR, "training_feature_stats.csv"))

    return best_name, best_metrics


if __name__ == "__main__":
    run_training()
