"""Per-employee retention recommendations via counterfactual simulation.

For a given employee, this doesn't guess generic advice — it takes the
employee's actual record, tweaks one company-actionable field at a time
(e.g. flips OverTime off, grants a training session, raises salary),
re-runs it through the *same trained pipeline*, and measures how much the
predicted attrition probability actually moves. Recommendations are ranked
by measured impact, not by a fixed rulebook.

Only fields a company can realistically act on are perturbed — things like
Age, Gender, or MaritalStatus are left untouched on purpose.
"""

from typing import Callable, NamedTuple


class Action(NamedTuple):
    feature: str
    label: str
    apply: Callable[[dict], dict]
    describe: Callable[[dict, dict], str]


def _clip(value, lo, hi):
    return max(lo, min(hi, value))


def _set(row: dict, **changes) -> dict:
    new_row = dict(row)
    new_row.update(changes)
    return new_row


ACTIONS = [
    Action(
        feature="OverTime",
        label="Eliminate mandatory overtime",
        apply=lambda row: _set(row, OverTime="No"),
        describe=lambda before, after: "Move off mandatory overtime — reduce workload or add headcount to the team.",
    ),
    Action(
        feature="BusinessTravel",
        label="Reduce business travel",
        apply=lambda row: _set(
            row,
            BusinessTravel={"Travel_Frequently": "Travel_Rarely", "Travel_Rarely": "Non-Travel"}.get(
                row["BusinessTravel"], row["BusinessTravel"]
            ),
        ),
        describe=lambda before, after: f"Cut business travel from {before['BusinessTravel']} to {after['BusinessTravel']}.",
    ),
    Action(
        feature="WorkLifeBalance",
        label="Improve work-life balance",
        apply=lambda row: _set(row, WorkLifeBalance=_clip(row["WorkLifeBalance"] + 1, 1, 4)),
        describe=lambda before, after: f"Improve work-life balance rating from {before['WorkLifeBalance']} to {after['WorkLifeBalance']} (flexible hours, workload review).",
    ),
    Action(
        feature="EnvironmentSatisfaction",
        label="Improve work environment",
        apply=lambda row: _set(row, EnvironmentSatisfaction=_clip(row["EnvironmentSatisfaction"] + 1, 1, 4)),
        describe=lambda before, after: f"Improve environment satisfaction from {before['EnvironmentSatisfaction']} to {after['EnvironmentSatisfaction']} (team, tooling, workspace).",
    ),
    Action(
        feature="JobSatisfaction",
        label="Improve role fit / job satisfaction",
        apply=lambda row: _set(row, JobSatisfaction=_clip(row["JobSatisfaction"] + 1, 1, 4)),
        describe=lambda before, after: f"Address job satisfaction via a role/scope conversation, from {before['JobSatisfaction']} to {after['JobSatisfaction']}.",
    ),
    Action(
        feature="RelationshipSatisfaction",
        label="Improve manager relationship",
        apply=lambda row: _set(row, RelationshipSatisfaction=_clip(row["RelationshipSatisfaction"] + 1, 1, 4)),
        describe=lambda before, after: f"Strengthen the manager relationship (1:1s, feedback) from {before['RelationshipSatisfaction']} to {after['RelationshipSatisfaction']}.",
    ),
    Action(
        feature="JobInvolvement",
        label="Increase job involvement",
        apply=lambda row: _set(row, JobInvolvement=_clip(row["JobInvolvement"] + 1, 1, 4)),
        describe=lambda before, after: f"Give more ownership/involvement in decisions, from {before['JobInvolvement']} to {after['JobInvolvement']}.",
    ),
    Action(
        feature="StockOptionLevel",
        label="Grant/increase equity",
        apply=lambda row: _set(row, StockOptionLevel=_clip(row["StockOptionLevel"] + 1, 0, 3)),
        describe=lambda before, after: f"Grant additional equity, stock option level {before['StockOptionLevel']} -> {after['StockOptionLevel']}.",
    ),
    Action(
        feature="TrainingTimesLastYear",
        label="Invest in training",
        apply=lambda row: _set(row, TrainingTimesLastYear=row["TrainingTimesLastYear"] + 2),
        describe=lambda before, after: f"Fund 2 more training sessions this year ({before['TrainingTimesLastYear']} -> {after['TrainingTimesLastYear']}).",
    ),
    Action(
        feature="PercentSalaryHike",
        label="Give an above-cycle raise",
        apply=lambda row: _set(row, PercentSalaryHike=row["PercentSalaryHike"] + 5),
        describe=lambda before, after: f"Increase the salary hike from {before['PercentSalaryHike']}% to {after['PercentSalaryHike']}%.",
    ),
    Action(
        feature="MonthlyIncome",
        label="Adjust base pay",
        apply=lambda row: _set(row, MonthlyIncome=int(round(row["MonthlyIncome"] * 1.10))),
        describe=lambda before, after: f"Adjust monthly income up 10%, from ${before['MonthlyIncome']:,} to ${after['MonthlyIncome']:,}.",
    ),
    Action(
        feature="YearsSinceLastPromotion",
        label="Promote now",
        apply=lambda row: _set(row, YearsSinceLastPromotion=0),
        describe=lambda before, after: f"Promote now — years since last promotion resets from {before['YearsSinceLastPromotion']} to 0.",
    ),
]


def generate_recommendations(pipeline, engineer_fn, base_row: dict, top_n: int = 5) -> dict:
    """Returns {baseline_probability, recommendations: [...]}, sorted by
    measured probability reduction (largest first). Only actions that
    actually reduce risk are included."""
    baseline_df = engineer_fn(base_row)
    baseline_proba = float(pipeline.predict_proba(baseline_df)[0, 1])

    scored = []
    for action in ACTIONS:
        new_row = action.apply(base_row)
        if new_row == base_row:
            continue  # action was a no-op for this employee (e.g. already Non-Travel)

        new_df = engineer_fn(new_row)
        new_proba = float(pipeline.predict_proba(new_df)[0, 1])
        reduction = baseline_proba - new_proba

        if reduction > 1e-6:
            scored.append(
                {
                    "feature": action.feature,
                    "label": action.label,
                    "description": action.describe(base_row, new_row),
                    "baseline_probability": round(baseline_proba, 4),
                    "new_probability": round(new_proba, 4),
                    "probability_reduction": round(reduction, 4),
                }
            )

    scored.sort(key=lambda r: r["probability_reduction"], reverse=True)

    return {
        "baseline_probability": round(baseline_proba, 4),
        "baseline_prediction": "Yes" if baseline_proba >= 0.5 else "No",
        "recommendations": scored[:top_n],
    }
