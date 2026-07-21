"""Validation tests for the data pipeline, model quality gate, and API contract.

Run with `pytest` from the repo root. The model-quality tests require
`models/model.joblib` to exist (run `python src/train.py` first) — they are
what CI runs after training to decide whether a build is good enough to ship.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_prep import ALL_FEATURES, TARGET, build_dataset  # noqa: E402
from evaluate import cross_check, manual_confusion_matrix, manual_metrics  # noqa: E402

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model.joblib")
MIN_RECALL = 0.55  # quality gate: catching departing employees matters more than raw accuracy
MIN_ACCURACY = 0.70


# ---------- data_prep ----------

def test_build_dataset_shape_and_no_nulls():
    X, y = build_dataset()
    assert len(X) == len(y)
    assert set(ALL_FEATURES).issubset(X.columns)
    assert X.isna().sum().sum() == 0
    assert set(y.unique()).issubset({0, 1})


def test_target_not_leaked_into_features():
    X, _ = build_dataset()
    assert TARGET not in X.columns


# ---------- evaluate (manual numpy metrics) ----------

def test_manual_confusion_matrix_matches_known_case():
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 1, 0])
    cm = manual_confusion_matrix(y_true, y_pred)
    # TN=2 (idx 2,5), FP=1 (idx 3), FN=1 (idx 1), TP=2 (idx 0,4)
    assert cm.tolist() == [[2, 1], [1, 2]]


def test_manual_metrics_match_hand_computed_values():
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 1, 0])
    m = manual_metrics(y_true, y_pred)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["accuracy"] == pytest.approx(4 / 6)


# ---------- model quality gate (requires a trained model artifact) ----------

@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="run src/train.py to produce models/model.joblib first")
def test_trained_model_meets_quality_gate():
    import joblib
    from sklearn.model_selection import train_test_split

    X, y = build_dataset()
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pipeline = joblib.load(MODEL_PATH)
    y_pred = pipeline.predict(X_test)
    y_score = pipeline.predict_proba(X_test)[:, 1]

    result = cross_check(y_test.values, y_pred, y_score)

    assert result["manual"]["recall"] >= MIN_RECALL, (
        f"Recall {result['manual']['recall']:.3f} below quality gate {MIN_RECALL}"
    )
    assert result["manual"]["accuracy"] >= MIN_ACCURACY


# ---------- API contract ----------

@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="run src/train.py to produce models/model.joblib first")
def test_predict_endpoint_returns_valid_schema(tmp_path, monkeypatch):
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(tmp_path / "prediction_log.csv"))
    monkeypatch.chdir(os.path.join(os.path.dirname(__file__), ".."))

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    import importlib

    import serve
    importlib.reload(serve)

    from fastapi.testclient import TestClient

    client = TestClient(serve.app)
    payload = serve.EmployeeRecord.model_config["json_schema_extra"]["example"]

    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["attrition_prediction"] in {"Yes", "No"}
    assert 0.0 <= body["attrition_probability"] <= 1.0


def test_health_endpoint():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    import serve
    from fastapi.testclient import TestClient

    client = TestClient(serve.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
