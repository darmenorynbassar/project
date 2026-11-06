"""Threshold selection and fairness analysis on the out-of-fold predictions.

Threshold (Section 3.5): for each CV fold k, the threshold is chosen on the
out-of-fold predictions of the other four folds so that recall on the at-risk
class reaches TARGET_RECALL, then applied to fold k. Fold k's own labels are
never used to pick its threshold.

Fairness (RQ3): with those thresholds, recall (true-positive rate), false-positive
rate, precision and ROC-AUC are compared across gender, age band, disability and
deprivation (IMD) band, with bootstrap 95% confidence intervals. The TPR gap
(max - min across groups) follows equality of opportunity (Hardt et al., 2016).

Run src/train.py first. Outputs contain aggregates only (no student ids).

Usage:
    python src/evaluate.py
"""
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from features import CUTOFF_WEEKS, PROCESSED, REPO

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = REPO / "results"
TARGET_RECALL = 0.75
MODELS = ["logreg", "random_forest", "xgboost"]
FAIRNESS_ATTRS = ["gender", "age_band", "disability", "imd_band"]
N_BOOT = 1000
SEED = 42


def threshold_for_recall(y, proba, target):
    """Highest threshold whose recall on (y, proba) is at least `target`."""
    pos = np.sort(proba[y == 1])
    return pos[int(np.floor((1 - target) * len(pos)))]


def cross_fitted_flags(df, model):
    proba, y, fold = df[f"proba_{model}"].values, df["at_risk"].values, df["fold"].values
    flags, thresholds = np.zeros(len(df), dtype=int), []
    for k in np.unique(fold):
        thr = threshold_for_recall(y[fold != k], proba[fold != k], TARGET_RECALL)
        flags[fold == k] = proba[fold == k] >= thr
        thresholds.append(thr)
    return flags, np.array(thresholds)


def rates(y, flags):
    tp = ((flags == 1) & (y == 1)).sum(axis=-1)
    fp = ((flags == 1) & (y == 0)).sum(axis=-1)
    pos, neg = (y == 1).sum(axis=-1), (y == 0).sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"recall": tp / pos, "fpr": fp / neg, "precision": tp / (tp + fp),
                "flag_rate": (flags == 1).mean(axis=-1)}


def bootstrap_ci(y, flags, rng):
    idx = rng.integers(0, len(y), size=(N_BOOT, len(y)))
    boot = rates(y[idx], flags[idx])
    return {f"{m}_ci_low": np.nanpercentile(v, 2.5) for m, v in boot.items()} | \
           {f"{m}_ci_high": np.nanpercentile(v, 97.5) for m, v in boot.items()}


def safe_auc(y, proba):
    return roc_auc_score(y, proba) if len(np.unique(y)) == 2 else np.nan


def plot_tradeoff(df, model, path, title):
    y, proba = df["at_risk"].values, df[f"proba_{model}"].values
    grid = np.linspace(0.05, 0.95, 91)
    r = rates(y[None, :], (proba[None, :] >= grid[:, None]).astype(int))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(grid, r["recall"], label="recall (at-risk)")
    ax.plot(grid, r["precision"], label="precision")
    ax.plot(grid, r["flag_rate"], label="share of students flagged", linestyle="--")
    ax.axhline(TARGET_RECALL, color="grey", linewidth=0.8, linestyle=":")
    ax.set_xlabel("decision threshold")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_fairness(fair, n, model, path):
    sub = fair[(fair["cutoff_weeks"] == n) & (fair["model"] == model)]
    fig, axes = plt.subplots(1, len(FAIRNESS_ATTRS), figsize=(15, 4.2),
                             gridspec_kw={"width_ratios": [2, 3, 2, 11]})
    for ax, attr in zip(axes, FAIRNESS_ATTRS):
        s = sub[sub["attribute"] == attr]
        groups = list(dict.fromkeys(s["group"]))
        x = np.arange(len(groups))
        for offset, (exp, marker) in zip([-0.15, 0.15], [("A", "o"), ("B", "s")]):
            e = s[s["experiment"] == exp].set_index("group").loc[groups]
            ax.errorbar(x + offset, e["recall"],
                        yerr=[e["recall"] - e["recall_ci_low"], e["recall_ci_high"] - e["recall"]],
                        fmt=marker, capsize=3, label=f"experiment {exp}")
        ax.axhline(TARGET_RECALL, color="grey", linewidth=0.8, linestyle=":")
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=8)
        ax.set_title(attr)
        ax.set_ylim(0.3, 1)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("recall on at-risk class (TPR)")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{model}, N = {n} weeks: recall by group (95% bootstrap CI), target {TARGET_RECALL}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    rng = np.random.default_rng(SEED)
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    thr_rows, fair_rows = [], []

    for n in CUTOFF_WEEKS:
        for exp in ["A", "B"]:
            df = pd.read_parquet(PROCESSED / f"oof_week{n}_{exp}.parquet")
            df[FAIRNESS_ATTRS] = df[FAIRNESS_ATTRS].fillna("missing")
            y = df["at_risk"].values
            for model in MODELS:
                flags, thresholds = cross_fitted_flags(df, model)
                key = {"cutoff_weeks": n, "experiment": exp, "model": model}
                overall = {m: float(v) for m, v in rates(y, flags).items()}
                thr_rows.append(key | {
                    "threshold_mean": thresholds.mean(), "threshold_sd": thresholds.std(),
                    **overall, **bootstrap_ci(y, flags, rng),
                    "f1": 2 * overall["precision"] * overall["recall"]
                          / (overall["precision"] + overall["recall"]),
                    "n_flagged": int(flags.sum()), "n_students": len(df)})

                for attr in FAIRNESS_ATTRS:
                    for group, g in df.groupby(attr, sort=True):
                        gi = g.index.values
                        yg, fg = y[gi], flags[gi]
                        fair_rows.append(key | {
                            "attribute": attr, "group": group, "n": len(gi),
                            "at_risk_share": yg.mean(),
                            **{m: float(v) for m, v in rates(yg, fg).items()},
                            **bootstrap_ci(yg, fg, rng),
                            "roc_auc": safe_auc(yg, g[f"proba_{model}"].values)})

            if n == 8 and exp == "A":
                plot_tradeoff(df, "xgboost", RESULTS / "figures" / "threshold_tradeoff_week8_A.png",
                              "XGBoost, N = 8 weeks, experiment A (out-of-fold)")

    thr = pd.DataFrame(thr_rows)
    thr.round(4).to_csv(RESULTS / "threshold_selection.csv", index=False)
    fair = pd.DataFrame(fair_rows)
    fair.round(4).to_csv(RESULTS / "fairness_by_group.csv", index=False)

    gaps = (fair.groupby(["cutoff_weeks", "experiment", "model", "attribute"])
            .agg(tpr_gap=("recall", lambda s: s.max() - s.min()),
                 fpr_gap=("fpr", lambda s: s.max() - s.min()),
                 lowest_tpr_group=("recall", lambda s: fair.loc[s.idxmin(), "group"]),
                 highest_fpr_group=("fpr", lambda s: fair.loc[s.idxmax(), "group"]))
            .reset_index())
    gaps.round(4).to_csv(RESULTS / "fairness_gaps.csv", index=False)

    for n in [8, 16]:
        plot_fairness(fair, n, "xgboost", RESULTS / "figures" / f"fairness_recall_week{n}.png")

    pd.set_option("display.width", 220)
    print(f"Cross-fitted threshold for target recall {TARGET_RECALL} (out-of-fold):")
    print(thr[["cutoff_weeks", "experiment", "model", "threshold_mean", "recall", "recall_ci_low",
               "recall_ci_high", "precision", "f1", "flag_rate"]].round(3).to_string(index=False))
    print("\nFairness gaps, XGBoost:")
    print(gaps[gaps["model"] == "xgboost"].drop(columns="model").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
