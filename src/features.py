"""Leakage-safe feature pipeline for OULAD.

For a cut-off of N weeks, every feature uses only information available
before day 7*N of the module presentation (OULAD dates are relative to the
module start). Students who unregistered before the cut-off are excluded,
because their outcome is already known at that point.

Usage:
    python src/features.py            # builds features for N = 4, 8, 12, 16
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
# README layout (data/raw) first, then the folder next to the repo
DATA_CANDIDATES = [REPO / "data" / "raw", REPO.parent / "anonymisedData"]
PROCESSED = REPO / "data" / "processed"  # git-ignored

KEYS = ["code_module", "code_presentation", "id_student"]
CUTOFF_WEEKS = [4, 8, 12, 16]
DEMOGRAPHIC_COLS = ["gender", "region", "highest_education", "imd_band", "age_band", "disability"]


def data_dir():
    for d in DATA_CANDIDATES:
        if (d / "studentInfo.csv").exists():
            return d
    raise FileNotFoundError(f"OULAD CSVs not found in any of: {DATA_CANDIDATES}")


def load_vle_daily():
    """studentVle joined with activity_type and summed per student/day/type.

    The raw file is ~450 MB, so the aggregated table is cached as Parquet.
    """
    cache = PROCESSED / "vle_daily.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    d = data_dir()
    vle = pd.read_csv(d / "vle.csv", usecols=["id_site", "activity_type"])
    sv = pd.read_csv(
        d / "studentVle.csv",
        dtype={"code_module": "category", "code_presentation": "category",
               "id_student": "int32", "id_site": "int32", "date": "int16", "sum_click": "int32"},
    )
    sv = sv.merge(vle, on="id_site", how="left")
    daily = (sv.groupby(KEYS + ["date", "activity_type"], observed=True)["sum_click"]
               .sum().reset_index())
    for c in ["code_module", "code_presentation", "activity_type"]:
        daily[c] = daily[c].astype(str)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(cache, index=False)
    return daily


def vle_features(daily, cutoff_day):
    v = daily[daily["date"] < cutoff_day]
    per_day = v.groupby(KEYS + ["date"])["sum_click"].sum().reset_index()

    g = per_day.groupby(KEYS)
    out = pd.DataFrame({
        "clicks_total": g["sum_click"].sum(),
        "active_days": g["date"].nunique(),
        "days_since_last_activity": cutoff_day - g["date"].max(),
        "clicks_last_2_weeks": per_day[per_day["date"] >= cutoff_day - 14].groupby(KEYS)["sum_click"].sum(),
        "clicks_before_start": per_day[per_day["date"] < 0].groupby(KEYS)["sum_click"].sum(),
    })

    # Weekly trend: slope of clicks per week over weeks 0..N-1 (pre-start clicks excluded)
    wk = per_day[per_day["date"] >= 0].assign(week=lambda x: x["date"] // 7)
    weekly = wk.groupby(KEYS + ["week"])["sum_click"].sum().unstack(fill_value=0)
    weekly = weekly.reindex(columns=range(cutoff_day // 7), fill_value=0)
    x = np.arange(weekly.shape[1]) - (weekly.shape[1] - 1) / 2
    out["weekly_click_slope"] = pd.Series((weekly.values @ x) / (x @ x), index=weekly.index)
    out["active_weeks"] = (weekly > 0).sum(axis=1)

    by_type = v.pivot_table(index=KEYS, columns="activity_type", values="sum_click",
                            aggfunc="sum", fill_value=0)
    by_type.columns = [f"clicks_{c}" for c in by_type.columns]
    return out.join(by_type, how="outer")


def assessment_features(students, assessments, sa, cutoff_day):
    """Non-exam assessments due before the cut-off, and submissions made before it."""
    due = assessments[(assessments["assessment_type"] != "Exam")
                      & (assessments["date"] < cutoff_day)]
    sub = sa[sa["date_submitted"] < cutoff_day].merge(
        assessments[["id_assessment", "code_module", "code_presentation", "date", "weight"]],
        on="id_assessment")
    sub["days_late"] = sub["date_submitted"] - sub["date"]

    g = sub.groupby(KEYS)
    out = pd.DataFrame({
        "n_submitted": g.size(),
        "mean_score": g["score"].mean(),
        "min_score": g["score"].min(),
        "n_late": g["days_late"].apply(lambda s: (s > 0).sum()),
        "mean_days_late": g["days_late"].mean(),
        "n_banked": g["is_banked"].sum(),
    })

    # Missed = due before the cut-off but not submitted before the cut-off
    pairs = students[KEYS].merge(due[["code_module", "code_presentation", "id_assessment"]],
                                 on=["code_module", "code_presentation"])
    pairs = pairs.merge(sub[["id_student", "id_assessment"]].assign(done=1),
                        on=["id_student", "id_assessment"], how="left")
    agg = pairs.groupby(KEYS).agg(n_due=("id_assessment", "size"), n_done=("done", "sum"))
    out = out.join(agg, how="outer")
    out["n_missed"] = out["n_due"].fillna(0) - out["n_done"].fillna(0)
    return out.drop(columns="n_done")


def build(cutoff_weeks, daily=None):
    d = data_dir()
    cutoff_day = 7 * cutoff_weeks
    info = pd.read_csv(d / "studentInfo.csv")
    reg = pd.read_csv(d / "studentRegistration.csv")
    assessments = pd.read_csv(d / "assessments.csv")
    sa = pd.read_csv(d / "studentAssessment.csv")
    if daily is None:
        daily = load_vle_daily()

    students = info.merge(reg, on=KEYS, how="left")
    n_all = len(students)
    # Exclude students whose withdrawal is already known at the cut-off
    students = students[~(students["date_unregistration"] < cutoff_day)].copy()
    n_excluded = n_all - len(students)

    students["at_risk"] = students["final_result"].isin(["Fail", "Withdrawn"]).astype(int)
    students["year"] = students["code_presentation"].str[:4].astype(int)

    feats = (students.set_index(KEYS)
             .join(vle_features(daily, cutoff_day))
             .join(assessment_features(students, assessments, sa, cutoff_day)))

    click_cols = [c for c in feats.columns if c.startswith("clicks_")] + \
                 ["active_days", "active_weeks", "n_submitted", "n_late",
                  "n_banked", "n_due", "n_missed", "weekly_click_slope"]
    feats[click_cols] = feats[click_cols].fillna(0)
    # No activity at all: treat as inactive for the whole window
    feats["days_since_last_activity"] = feats["days_since_last_activity"].fillna(cutoff_day + 1)
    # Clicks relative to the median student in the same presentation (peers' data up to the cut-off)
    med = feats.groupby(["code_module", "code_presentation"])["clicks_total"].transform("median")
    feats["clicks_rel_to_median"] = feats["clicks_total"] / med.replace(0, np.nan)

    feats = feats.drop(columns=["date_unregistration"]).reset_index()
    feats.attrs["n_excluded"] = n_excluded
    return feats


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    daily = load_vle_daily()
    for n in CUTOFF_WEEKS:
        f = build(n, daily)
        f.to_parquet(PROCESSED / f"features_week{n}.parquet", index=False)
        print(f"N={n:>2}: {len(f):>6} students kept, {f.attrs['n_excluded']:>5} excluded "
              f"(unregistered before cut-off), at-risk share {f['at_risk'].mean():.3f}, "
              f"{f.shape[1]} columns")


if __name__ == "__main__":
    main()
