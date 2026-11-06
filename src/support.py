"""Support layer: risk list, explanations, advisor letter, mock meeting slots.

Every number shown to a user comes from here, never from the language model.
Nothing is sent anywhere: the action log is a local, simulated record (UC5).
Run src/build_support.py first.
"""
import json
from datetime import datetime
from functools import lru_cache

import pandas as pd

from features import CUTOFF_WEEKS, PROCESSED, REPO

ACTION_LOG = PROCESSED / "action_log.csv"  # git-ignored
METRICS = REPO / "results" / "threshold_out_of_time.csv"
LEVEL_ORDER = {"High": 0, "Medium": 1, "Low": 2}
MOCK_SLOTS = [f"Week +1, {day} {time}" for day in ["Monday", "Tuesday", "Thursday"]
              for time in ["10:00", "14:00"]]

FEATURE_LABELS = {
    "mean_score": "average assessment score",
    "min_score": "lowest assessment score",
    "n_submitted": "assessments submitted",
    "n_missed": "assessments missed",
    "n_due": "assessments due so far",
    "n_late": "late submissions",
    "mean_days_late": "average days late (negative means early)",
    "n_banked": "banked assessments",
    "clicks_total": "total VLE clicks",
    "clicks_rel_to_median": "VLE clicks relative to the median student (1.0 = median)",
    "clicks_last_2_weeks": "VLE clicks in the last 2 weeks",
    "clicks_before_start": "VLE clicks before the module started",
    "active_days": "days active on the VLE",
    "active_weeks": "weeks active on the VLE",
    "days_since_last_activity": "days since last VLE activity",
    "weekly_click_slope": "weekly change in VLE clicks",
    "date_registration": "registration day relative to module start",
    "num_of_prev_attempts": "previous attempts at this module",
    "studied_credits": "credits being studied",
}


def label(feature):
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    if feature.startswith("clicks_"):
        return f"clicks on '{feature[len('clicks_'):]}' VLE activities"
    return feature


@lru_cache(maxsize=None)
def load(week):
    if week not in CUTOFF_WEEKS:
        raise ValueError(f"week must be one of {CUTOFF_WEEKS}")
    return pd.read_parquet(PROCESSED / f"support_week{week}.parquet")


def modules(week):
    df = load(week)
    g = df.groupby(["code_module", "code_presentation"])
    table = g.agg(students=("id_student", "size"), flagged=("flagged", "sum")).reset_index()
    return table.to_dict("records")


def risk_list(week, module=None, presentation=None, limit=10):
    df = load(week)
    df = df[df["flagged"]]
    if module:
        df = df[df["code_module"] == module.upper()]
    if presentation:
        df = df[df["code_presentation"] == presentation.upper()]
    df = df.sort_values("risk_score", ascending=False)
    cols = ["id_student", "code_module", "code_presentation", "risk_level", "risk_score",
            "mean_score", "n_missed", "days_since_last_activity"]
    return {"week": week, "total_flagged": len(df),
            "students": df.head(limit)[cols].round(3).astype(object)
                          .where(lambda d: d.notna(), None).to_dict("records")}


def explain(student_id, week, module=None):
    """One record per module the student is registered on at this cut-off."""
    df = load(week)
    df = df[df["id_student"] == int(student_id)]
    if module:
        df = df[df["code_module"] == module.upper()]
    if df.empty:
        return {"found": False, "week": week,
                "note": "Not in the 2014 test students at this cut-off "
                        "(wrong id, or unregistered before the cut-off)."}
    records = []
    for r in df.to_dict("records"):
        factors = [{"factor": label(f["feature"]), "value": f["value"],
                    "effect": "raises risk" if f["shap"] > 0 else "lowers risk"}
                   for f in json.loads(r["factors"])]
        records.append({"student_id": int(r["id_student"]), "module": r["code_module"],
                        "presentation": r["code_presentation"], "week": week,
                        "risk_level": r["risk_level"], "risk_score": float(r["risk_score"]),
                        "flagged": bool(r["flagged"]), "top_factors": factors})
    return {"found": True, "records": records}


def draft_letter(student_id, week, module=None):
    result = explain(student_id, week, module)
    if not result["found"]:
        return result
    if len(result["records"]) > 1:
        return {"found": True, "needs_module": [r["module"] for r in result["records"]]}
    r = result["records"][0]
    lines = "\n".join(f"- {f['factor'][0].upper()}{f['factor'][1:]}: {f['value']} ({f['effect']})"
                      for f in r["top_factors"])
    letter = (
        f"Dear Advisor,\n\n"
        f"The early-warning model has marked student {r['student_id']} "
        f"({r['module']} {r['presentation']}) as {r['risk_level'].lower()} risk at week {week}.\n\n"
        f"The main factors behind this result are:\n{lines}\n\n"
        f"This is decision support, not a judgement about the student, and the model can be "
        f"wrong. Please review the case and, if you agree, consider offering a supportive "
        f"meeting.\n\nKind regards,\nAcademic Support Team"
    )
    return {"found": True, "student_id": r["student_id"], "module": r["module"], "letter": letter}


def booked_slots():
    if not ACTION_LOG.exists():
        return set()
    return set(pd.read_csv(ACTION_LOG)["slot"].dropna())


def meeting_slots():
    return [s for s in MOCK_SLOTS if s not in booked_slots()]


def model_info(week):
    row = pd.read_csv(METRICS).set_index("cutoff_weeks").loc[week]
    return {"model": "XGBoost, no demographic inputs, trained on 2013, tested on unseen 2014 students",
            "week": week, "threshold": float(row["threshold"]),
            "recall": float(row["recall"]), "precision": float(row["precision"]),
            "share_of_students_flagged": float(row["flag_rate"]),
            "note": "Risk score is a ranking score, not a calibrated probability. "
                    "A human reviews every case; nothing is sent automatically."}


def record_action(student_id, module, week, letter, slot, outcome):
    """UC5, simulated. Called only from a human button press, never by the language model."""
    row = pd.DataFrame([{"time": datetime.now().isoformat(timespec="seconds"),
                         "student_id": student_id, "module": module, "week": week,
                         "slot": slot, "outcome": outcome, "letter": letter}])
    row.to_csv(ACTION_LOG, mode="a", header=not ACTION_LOG.exists(), index=False)
