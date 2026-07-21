"""Load, clean, and feature-engineer the IBM HR Employee Attrition dataset."""

import numpy as np
import pandas as pd

RAW_PATH = "data/employee_attrition.csv"

TARGET = "Attrition"

# Columns that carry no signal: single constant value across all rows, or a
# row identifier with no predictive meaning.
CONSTANT_COLUMNS = ["EmployeeCount", "StandardHours", "Over18"]
ID_COLUMNS = ["EmployeeNumber"]

CATEGORICAL_FEATURES = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
    "TenureBucket",
]

NUMERIC_FEATURES = [
    "Age",
    "DailyRate",
    "DistanceFromHome",
    "Education",
    "EnvironmentSatisfaction",
    "HourlyRate",
    "JobInvolvement",
    "JobLevel",
    "JobSatisfaction",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
    "IncomePerJobLevel",
    "PromotionGapRatio",
    "ManagerTenureRatio",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop dead-weight columns and encode the target as 0/1."""
    df = df.drop(columns=CONSTANT_COLUMNS + ID_COLUMNS, errors="ignore").copy()
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


def _tenure_bucket(years: pd.Series) -> pd.Series:
    bins = [-np.inf, 1, 3, 7, 15, np.inf]
    labels = ["<1yr", "1-3yr", "3-7yr", "7-15yr", "15yr+"]
    return pd.cut(years, bins=bins, labels=labels).astype(str)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized numpy/pandas feature engineering on top of the cleaned frame."""
    df = df.copy()

    df["TenureBucket"] = _tenure_bucket(df["YearsAtCompany"])

    # Income relative to job level: flags employees paid out of step with peers.
    df["IncomePerJobLevel"] = df["MonthlyIncome"] / df["JobLevel"].replace(0, 1)

    # How much of an employee's tenure has passed since their last promotion.
    df["PromotionGapRatio"] = np.where(
        df["YearsAtCompany"] > 0,
        df["YearsSinceLastPromotion"] / df["YearsAtCompany"],
        0.0,
    )

    # Share of company tenure spent under the current manager (relationship stability).
    df["ManagerTenureRatio"] = np.where(
        df["YearsAtCompany"] > 0,
        df["YearsWithCurrManager"] / df["YearsAtCompany"],
        0.0,
    )

    return df


def get_feature_target(df: pd.DataFrame):
    X = df[ALL_FEATURES]
    y = df[TARGET]
    return X, y


def build_dataset(path: str = RAW_PATH):
    """End-to-end: raw CSV -> (X, y) ready for a train/test split."""
    raw = load_raw(path)
    cleaned = clean(raw)
    engineered = engineer_features(cleaned)
    return get_feature_target(engineered)


if __name__ == "__main__":
    X, y = build_dataset()
    print(f"Feature matrix: {X.shape}, target balance: {y.mean():.3f} attrition rate")
    print(X.dtypes)
