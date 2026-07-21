"""Confusion matrix and classification metrics computed by hand from raw
numpy arrays, then cross-checked against scikit-learn's implementations.

This exists to prove the metrics are understood, not just imported.
"""

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def manual_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Returns [[TN, FP], [FN, TP]], matching sklearn's layout."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return np.array([[tn, fp], [fn, tp]])


def manual_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tn, fp, fn, tp = manual_confusion_matrix(y_true, y_pred).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def manual_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Rank-based (Mann-Whitney U) AUC — no threshold sweep required."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)

    ranks = rankdata(y_score, method="average")  # average ranks so tied proba scores don't bias the sum
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    sum_ranks_pos = np.sum(ranks[y_true == 1])
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def cross_check(y_true, y_pred, y_score) -> dict:
    """Compute metrics both ways and assert they agree within floating-point tolerance."""
    manual = manual_metrics(y_true, y_pred)
    manual["roc_auc"] = manual_roc_auc(y_true, y_score)

    sklearn_metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_score),
    }
    sklearn_tn, sklearn_fp, sklearn_fn, sklearn_tp = confusion_matrix(y_true, y_pred).ravel()

    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        assert np.isclose(manual[key], sklearn_metrics[key], atol=1e-6), (
            f"Mismatch on {key}: manual={manual[key]} sklearn={sklearn_metrics[key]}"
        )
    assert (manual["tn"], manual["fp"], manual["fn"], manual["tp"]) == (
        sklearn_tn,
        sklearn_fp,
        sklearn_fn,
        sklearn_tp,
    )

    return {"manual": manual, "sklearn": sklearn_metrics}


if __name__ == "__main__":
    import joblib
    from sklearn.model_selection import train_test_split

    from data_prep import build_dataset

    X, y = build_dataset()
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pipeline = joblib.load("models/model.joblib")
    y_pred = pipeline.predict(X_test)
    y_score = pipeline.predict_proba(X_test)[:, 1]

    result = cross_check(y_test.values, y_pred, y_score)
    print("Manual metrics: ", result["manual"])
    print("Sklearn metrics:", result["sklearn"])
    print("\nAll metrics match within tolerance.")
