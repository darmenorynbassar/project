"""Train and compare at-risk models on the OULAD feature tables.

For every cut-off N and both feature sets (A: behaviour + registration,
B: A + demographics) this script evaluates a majority-class baseline,
logistic regression, random forest and XGBoost with

  * 5-fold StratifiedGroupKFold grouped by student (no student in both train and test)
  * an out-of-time check: train on 2013 presentations, test on 2014 presentations,
    dropping test students who also appear in training

All preprocessing lives inside the sklearn Pipeline, so it is fitted on
training folds only. Run src/features.py first.

Usage:
    python src/train.py
"""
import warnings

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from features import CUTOFF_WEEKS, DEMOGRAPHIC_COLS, KEYS, PROCESSED, REPO

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = REPO / "results"
MODELS = REPO / "models"  # git-ignored: trained on the dataset
SEED = 42

# Never used as features: identifiers, outcome and its derivatives
NON_FEATURES = KEYS + ["final_result", "at_risk", "year"]


def feature_columns(df, experiment):
    cols = [c for c in df.columns if c not in NON_FEATURES and c not in DEMOGRAPHIC_COLS]
    if experiment == "B":
        cols += DEMOGRAPHIC_COLS
    categorical = [c for c in cols if c in DEMOGRAPHIC_COLS]
    numeric = [c for c in cols if c not in categorical]
    return numeric, categorical


def make_models(pos_weight):
    return {
        "baseline": DummyClassifier(strategy="most_frequent"),
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced",
            n_jobs=-1, random_state=SEED),
        "xgboost": XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=5, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=pos_weight, eval_metric="logloss",
            n_jobs=-1, random_state=SEED),
    }


def make_pipeline(model, numeric, categorical):
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def scores(y, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    single_class = len(np.unique(y)) < 2
    return {
        "recall": recall_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": np.nan if single_class else roc_auc_score(y, proba),
        "pr_auc": np.nan if single_class else average_precision_score(y, proba),
    }


def cross_validate(df, numeric, categorical):
    X, y, groups = df[numeric + categorical], df["at_risk"].values, df["id_student"].values
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, oof = [], {}
    for name in make_models(1).keys():
        oof[name] = np.zeros(len(df))
    oof["fold"] = np.zeros(len(df), dtype=int)
    for fold, (tr, te) in enumerate(cv.split(X, y, groups)):
        oof["fold"][te] = fold
        pos_weight = (y[tr] == 0).sum() / (y[tr] == 1).sum()
        for name, model in make_models(pos_weight).items():
            pipe = make_pipeline(model, numeric, categorical).fit(X.iloc[tr], y[tr])
            proba = pipe.predict_proba(X.iloc[te])[:, 1]
            oof[name][te] = proba
            rows.append({"model": name, "fold": fold, **scores(y[te], proba)})
    return pd.DataFrame(rows), oof


def out_of_time(df, numeric, categorical):
    train = df[df["year"] == 2013]
    test = df[(df["year"] == 2014) & ~df["id_student"].isin(train["id_student"])]
    y_tr, y_te = train["at_risk"].values, test["at_risk"].values
    pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()
    rows, fitted = [], {}
    for name, model in make_models(pos_weight).items():
        pipe = make_pipeline(model, numeric, categorical).fit(train[numeric + categorical], y_tr)
        proba = pipe.predict_proba(test[numeric + categorical])[:, 1]
        rows.append({"model": name, "n_train": len(train), "n_test": len(test),
                     **scores(y_te, proba)})
        fitted[name] = pipe
    return pd.DataFrame(rows), fitted


def shap_summary(pipe, df, numeric, categorical, path, title):
    import shap
    pre, model = pipe.named_steps["pre"], pipe.named_steps["model"]
    sample = df[numeric + categorical].sample(n=min(3000, len(df)), random_state=SEED)
    X = pre.transform(sample)
    X = X.toarray() if hasattr(X, "toarray") else X
    names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
    values = shap.TreeExplainer(model).shap_values(X)
    plt.figure()
    shap.summary_plot(values, X, feature_names=names, max_display=15, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")


def plot_earliness(summary, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True)
    for ax, metric in zip(axes, ["recall", "f1", "roc_auc"]):
        for (model, exp), g in summary[summary["model"] != "baseline"].groupby(["model", "experiment"]):
            g = g.sort_values("cutoff_weeks")
            ax.errorbar(g["cutoff_weeks"], g[f"{metric}_mean"], yerr=g[f"{metric}_std"],
                        marker="o", capsize=3, label=f"{model} ({exp})",
                        linestyle="-" if exp == "A" else "--")
        ax.set_title(metric.replace("_", "-").upper())
        ax.set_xlabel("cut-off week N")
        ax.set_xticks(CUTOFF_WEEKS)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("At-risk class, 5-fold group CV (mean ± SD)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    cv_rows, oot_rows = [], []

    for n in CUTOFF_WEEKS:
        df = pd.read_parquet(PROCESSED / f"features_week{n}.parquet")
        for exp in ["A", "B"]:
            numeric, categorical = feature_columns(df, exp)
            print(f"N={n:>2} experiment {exp}: {len(df)} students, "
                  f"{len(numeric)} numeric + {len(categorical)} categorical features")

            cv, oof = cross_validate(df, numeric, categorical)
            cv_rows.append(cv.assign(cutoff_weeks=n, experiment=exp))
            # Out-of-fold predictions for the fairness analysis (contain student ids: stays local)
            preds = df[KEYS + DEMOGRAPHIC_COLS + ["at_risk"]].copy()
            preds["fold"] = oof.pop("fold")
            for name, p in oof.items():
                preds[f"proba_{name}"] = p
            preds.to_parquet(PROCESSED / f"oof_week{n}_{exp}.parquet", index=False)

            oot, fitted = out_of_time(df, numeric, categorical)
            oot_rows.append(oot.assign(cutoff_weeks=n, experiment=exp))
            joblib.dump({"pipeline": fitted["xgboost"], "numeric": numeric,
                         "categorical": categorical, "cutoff_weeks": n},
                        MODELS / f"xgboost_week{n}_{exp}.joblib")

            if exp == "A":
                shap_summary(fitted["xgboost"], df, numeric, categorical,
                             RESULTS / "figures" / f"shap_xgboost_week{n}_A.png",
                             f"XGBoost, N = {n} weeks, experiment A")

    cv_all = pd.concat(cv_rows)
    cv_all.to_csv(RESULTS / "metrics_cv_folds.csv", index=False)
    metrics = ["recall", "precision", "f1", "roc_auc", "pr_auc"]
    summary = (cv_all.groupby(["cutoff_weeks", "experiment", "model"])[metrics]
               .agg(["mean", "std"]))
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()
    summary.round(4).to_csv(RESULTS / "metrics_cv_summary.csv", index=False)
    oot_all = pd.concat(oot_rows)
    oot_all.round(4).to_csv(RESULTS / "metrics_out_of_time.csv", index=False)
    plot_earliness(summary, RESULTS / "figures" / "earliness.png")

    pd.set_option("display.width", 200)
    print("\n5-fold group CV, at-risk class (mean):")
    print(summary.pivot_table(index=["cutoff_weeks", "model"], columns="experiment",
                              values=["recall_mean", "f1_mean", "roc_auc_mean"]).round(3))
    print("\nOut-of-time (train 2013 -> test 2014):")
    print(oot_all.pivot_table(index=["cutoff_weeks", "model"], columns="experiment",
                              values=["recall", "f1", "roc_auc"]).round(3))


if __name__ == "__main__":
    main()
