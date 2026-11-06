"""Precompute what the support layer shows, for each cut-off week.

The models saved by train.py are trained on 2013 presentations only, so the
prototype works on 2014 students the model has never seen. Experiment A is
used: no demographic inputs (see the fairness results in Section 4.2).

The threshold for recall 0.75 is chosen on 2013 only, with group CV inside the
training year, and then applied once to 2014. This gives an honest
out-of-time estimate of recall, precision and alert volume.

Run src/train.py first.

Usage:
    python src/build_support.py
"""
import json

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import StratifiedGroupKFold

from evaluate import TARGET_RECALL, rates, threshold_for_recall
from features import CUTOFF_WEEKS, PROCESSED
from train import MODELS, RESULTS, SEED, make_models, make_pipeline

HIGH_RISK_SCORE = 0.75
N_FACTORS = 3
FACT_COLS = ["mean_score", "n_submitted", "n_missed", "n_late", "clicks_total",
             "clicks_rel_to_median", "active_days", "days_since_last_activity"]


def threshold_from_2013(train, numeric):
    """Out-of-fold XGBoost scores within 2013, then the threshold for the target recall."""
    X, y = train[numeric], train["at_risk"].values
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train))
    for tr, te in cv.split(X, y, train["id_student"].values):
        pos_weight = (y[tr] == 0).sum() / (y[tr] == 1).sum()
        pipe = make_pipeline(make_models(pos_weight)["xgboost"], numeric, [])
        oof[te] = pipe.fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
    return float(threshold_for_recall(y, oof, TARGET_RECALL))


def top_factors(shap_row, raw_row, numeric, flagged):
    """Features pushing the score up for flagged students, down for the others."""
    order = np.argsort(-shap_row if flagged else shap_row)[:N_FACTORS]
    return [{"feature": numeric[i], "value": round(float(raw_row[i]), 2),
             "shap": round(float(shap_row[i]), 3)}
            for i in order if not np.isnan(raw_row[i])]


def build(n):
    df = pd.read_parquet(PROCESSED / f"features_week{n}.parquet")
    saved = joblib.load(MODELS / f"xgboost_week{n}_A.joblib")
    pipe, numeric = saved["pipeline"], saved["numeric"]

    train = df[df["year"] == 2013]
    test = df[(df["year"] == 2014) & ~df["id_student"].isin(train["id_student"])].reset_index(drop=True)
    threshold = threshold_from_2013(train, numeric)

    score = pipe.predict_proba(test[numeric])[:, 1]
    flagged = score >= threshold
    values = shap.TreeExplainer(pipe.named_steps["model"]).shap_values(
        pipe.named_steps["pre"].transform(test[numeric]))
    raw = test[numeric].to_numpy(dtype=float)

    out = test[["code_module", "code_presentation", "id_student", "at_risk"] + FACT_COLS].copy()
    out["risk_score"] = score.round(4)
    out["flagged"] = flagged
    out["risk_level"] = np.where(score >= max(HIGH_RISK_SCORE, threshold), "High",
                                 np.where(flagged, "Medium", "Low"))
    out["factors"] = [json.dumps(top_factors(values[i], raw[i], numeric, flagged[i]))
                      for i in range(len(test))]
    out.to_parquet(PROCESSED / f"support_week{n}.parquet", index=False)

    y = test["at_risk"].values
    m = rates(y, flagged.astype(int))
    return {"cutoff_weeks": n, "threshold": round(threshold, 4), "n_students": len(test),
            "n_flagged": int(flagged.sum()), **{k: round(float(v), 4) for k, v in m.items()}}


def main():
    rows = [build(n) for n in CUTOFF_WEEKS]
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "threshold_out_of_time.csv", index=False)
    print(f"Threshold chosen on 2013 for recall {TARGET_RECALL}, applied to 2014 (XGBoost, experiment A):")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
