"""Generate the CSVs Power BI reads from: cleaned/engineered data for the EDA
page, and model performance artifacts (confusion matrix, metrics, feature
importance) for the model performance page. Run after train.py.
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data_prep import TARGET, build_dataset, clean, engineer_features, load_raw
from evaluate import manual_confusion_matrix, manual_metrics, manual_roc_auc

EXPORT_DIR = "reports/exports"
MODEL_PATH = "models/model.joblib"


def export_clean_data():
    raw = load_raw()
    df = engineer_features(clean(raw))
    df["Attrition"] = df[TARGET].map({1: "Yes", 0: "No"})
    df.to_csv(os.path.join(EXPORT_DIR, "employee_data_clean.csv"), index=False)
    print(f"Wrote {len(df)} rows -> employee_data_clean.csv")


def export_model_performance():
    if not os.path.exists(MODEL_PATH):
        print("No trained model found, skipping model performance export. Run train.py first.")
        return

    X, y = build_dataset()
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pipeline = joblib.load(MODEL_PATH)
    y_pred = pipeline.predict(X_test)
    y_score = pipeline.predict_proba(X_test)[:, 1]

    # Confusion matrix, long format (ideal for a Power BI matrix visual).
    cm = manual_confusion_matrix(y_test.values, y_pred)
    cm_df = pd.DataFrame(
        [
            {"actual": "No", "predicted": "No", "count": int(cm[0, 0])},
            {"actual": "No", "predicted": "Yes", "count": int(cm[0, 1])},
            {"actual": "Yes", "predicted": "No", "count": int(cm[1, 0])},
            {"actual": "Yes", "predicted": "Yes", "count": int(cm[1, 1])},
        ]
    )
    cm_df.to_csv(os.path.join(EXPORT_DIR, "confusion_matrix.csv"), index=False)

    # Scalar metrics, long format (metric, value) so DAX can pick them up as a table.
    metrics = manual_metrics(y_test.values, y_pred)
    metrics["roc_auc"] = manual_roc_auc(y_test.values, y_score)
    metrics_df = pd.DataFrame(
        [{"metric": k, "value": v} for k, v in metrics.items() if k in
         ("accuracy", "precision", "recall", "f1", "roc_auc")]
    )
    metrics_df.to_csv(os.path.join(EXPORT_DIR, "model_metrics.csv"), index=False)

    # ROC curve points, computed by hand by sweeping thresholds over the sorted scores.
    thresholds = np.linspace(0, 1, 101)
    roc_rows = []
    y_true_arr = y_test.values
    for t in thresholds:
        preds = (y_score >= t).astype(int)
        tp = int(np.sum((y_true_arr == 1) & (preds == 1)))
        fn = int(np.sum((y_true_arr == 1) & (preds == 0)))
        fp = int(np.sum((y_true_arr == 0) & (preds == 1)))
        tn = int(np.sum((y_true_arr == 0) & (preds == 0)))
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        roc_rows.append({"threshold": round(float(t), 2), "tpr": tpr, "fpr": fpr})
    pd.DataFrame(roc_rows).to_csv(os.path.join(EXPORT_DIR, "roc_curve.csv"), index=False)

    # Feature importance / coefficients, whichever the winning model exposes.
    model = pipeline.named_steps["model"]
    preprocess = pipeline.named_steps["preprocess"]
    feature_names = preprocess.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        importances = None

    if importances is not None:
        fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        fi_df = fi_df.sort_values("importance", ascending=False)
        fi_df.to_csv(os.path.join(EXPORT_DIR, "feature_importance.csv"), index=False)

    print("Wrote confusion_matrix.csv, model_metrics.csv, roc_curve.csv, feature_importance.csv")


if __name__ == "__main__":
    os.makedirs(EXPORT_DIR, exist_ok=True)
    export_clean_data()
    export_model_performance()
