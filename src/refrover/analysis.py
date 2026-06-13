"""
Analysis of the grid (PLAN.md §7, build step 6) — the primary deliverable.

The mean-over-samples leaderboard (`grid.headline`) hides the point. §7 asks the
sharper question:

    What cheaply-measurable property of a sample predicts which selection method
    is best *for that sample*?

If we can answer it, we get an adaptive meta-selector. These functions turn the
grid's per-sample score table (and the per-sample feature table from
`features.py`) into that analysis:

  - winning_method / win_rates      — which rule wins per sample, and how often
  - method_advantage                — score(rule) − score(baseline) per sample
                                       (the §7.2 regression target)
  - winner_feature_contrast         — how per-sample features differ by winner
                                       (the interpretable "when X, rule Y wins")
  - fit_winner_rule                  — an explicit human-readable decision rule
                                       (optional; needs scikit-learn)
  - divergence                       — §7.3 overlap: where methods disagree (and
                                       so the choice — and the features — matter)

These are dataset-agnostic: they take the grid/feature tables and return tidy
frames. The mouse-specific wiring lives in
`data/mouse_rewilding/run_grid_analysis.py`.
"""

from __future__ import annotations

import pandas as pd

DEFAULT_METRIC = "tiered_axis_count"  # the §6 default, most MAG-shaped proxy
#: all_vs_all is the O(n²) ceiling, not a deployable selection method — exclude it
#: from "which method wins" comparisons by default.
DEFAULT_EXCLUDE = ("all_vs_all",)


def winning_method(
    scores: pd.DataFrame,
    *,
    metric: str = DEFAULT_METRIC,
    exclude: tuple[str, ...] = DEFAULT_EXCLUDE,
) -> pd.DataFrame:
    """
    The best rule for each (feature_space, k, sample) by `metric`.

    Ties resolve to the first rule encountered (stable order). Returns tidy
    columns [feature_space, k, sample_id, winner, <metric>]. ``all_vs_all`` (and
    anything else in `exclude`) is dropped — we want the best *deployable* rule.
    """
    df = scores[~scores["rule"].isin(exclude)]
    idx = df.groupby(["feature_space", "k", "sample_id"])[metric].idxmax()
    winners = df.loc[idx, ["feature_space", "k", "sample_id", "rule", metric]]
    return winners.rename(columns={"rule": "winner"}).reset_index(drop=True)


def win_rates(winners: pd.DataFrame) -> pd.DataFrame:
    """
    Fraction of samples each rule wins, per (feature_space, k).

    More honest than a mean score (§7): "css wins 60% of samples" beats "css is
    higher by 3% on average". Sorted by win_rate desc within each group.
    """
    counts = winners.groupby(["feature_space", "k", "winner"]).size().rename("n_wins")
    totals = winners.groupby(["feature_space", "k"]).size().rename("n_samples")
    out = counts.reset_index().merge(totals.reset_index(), on=["feature_space", "k"])
    out["win_rate"] = out["n_wins"] / out["n_samples"]
    return out.sort_values(
        ["feature_space", "k", "win_rate"], ascending=[True, True, False],
        ignore_index=True,
    )


def method_advantage(
    scores: pd.DataFrame,
    *,
    metric: str = DEFAULT_METRIC,
    baseline: str = "random",
    exclude: tuple[str, ...] = DEFAULT_EXCLUDE,
) -> pd.DataFrame:
    """
    Per-sample advantage of each rule over `baseline`: score(rule) − score(baseline).

    This is the §7.2 regression target (`advantage ~ features`) — a continuous,
    per-sample signal of how much a rule helps *this* sample. Returns tidy columns
    [feature_space, rule, k, sample_id, advantage]. The baseline and `exclude`
    rules are themselves dropped from the rows.
    """
    base = (
        scores[scores["rule"] == baseline]
        [["feature_space", "k", "sample_id", metric]]
        .rename(columns={metric: "_baseline"})
    )
    drop = set(exclude) | {baseline}
    df = scores[~scores["rule"].isin(drop)].merge(
        base, on=["feature_space", "k", "sample_id"], how="left"
    )
    df["advantage"] = df[metric] - df["_baseline"]
    return df[["feature_space", "rule", "k", "sample_id", "advantage"]].reset_index(drop=True)


def winner_feature_contrast(
    winners: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Mean per-sample feature value grouped by which rule won (§7.2, interpretable).

    Joins winners to the per-sample feature table and averages each feature within
    each winner group, per (feature_space, k). Reading across a row tells the
    story: "where css wins, samples have high species_distinctness and low
    evenness; where archetype wins, they're diverse and even."

    `features` is indexed by sample_id (as returned by `features.feature_table`).
    """
    feat = features.reset_index()
    feat_cols = list(features.columns)
    merged = winners.merge(feat, on="sample_id", how="left")
    return (
        merged.groupby(["feature_space", "k", "winner"])[feat_cols]
        .mean()
        .reset_index()
    )


def divergence(overlap: pd.DataFrame) -> pd.DataFrame:
    """
    Per-sample mean pairwise rule agreement (§7.3 overlap analysis).

    Averages the pairwise Jaccard of chosen reference sets over all rule pairs,
    per (feature_space, k, sample). Low ``mean_agreement`` = the rules diverge for
    that sample, so the choice of method matters there — these are the samples
    where the per-sample features should arbitrate. Sorted most-divergent first.
    """
    out = (
        overlap.groupby(["feature_space", "k", "sample_id"])["jaccard"]
        .mean()
        .rename("mean_agreement")
        .reset_index()
    )
    return out.sort_values("mean_agreement", ascending=True, ignore_index=True)


def fit_winner_rule(
    winners: pd.DataFrame,
    features: pd.DataFrame,
    *,
    feature_space: str,
    k: int,
    max_depth: int = 3,
    cv: int = 5,
) -> dict:
    """
    Fit a shallow decision tree predicting the winning rule from per-sample
    features, for one (feature_space, k) — the §7.2 "human-readable rule".

    The interpretable rule text comes from a tree fit on all samples, but the
    headline number is the **cross-validated** accuracy, compared against a
    majority-class baseline: a feature→method rule is only real if it beats just
    always guessing the most common winner. ``train_accuracy`` is reported too, so
    the train-vs-CV gap exposes overfitting (small per-(fs,k) sample counts).

    Returns ``{"rules", "train_accuracy", "cv_accuracy", "cv_std",
    "baseline_accuracy", "lift", "cv_folds", "n_samples", "classes"}``.
    ``cv_accuracy``/``cv_std``/``lift`` are ``None`` when stratified CV is
    infeasible (a winning class with too few samples); ``cv_folds`` records how
    many folds were actually used (capped at the smallest class's size).

    Requires scikit-learn (an existing dependency); raises ImportError if absent.
    """
    try:
        from collections import Counter

        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.tree import DecisionTreeClassifier, export_text
    except ImportError as exc:  # pragma: no cover - optional
        raise ImportError(
            "fit_winner_rule requires scikit-learn: pip install scikit-learn"
        ) from exc

    w = winners[(winners["feature_space"] == feature_space) & (winners["k"] == k)]
    X = features.loc[w["sample_id"]]
    y = w["winner"].to_numpy()
    classes = sorted(set(y))
    n = int(len(y))

    if len(classes) < 2:
        return {
            "rules": f"(only one winner: {classes[0] if classes else 'none'})",
            "train_accuracy": 1.0 if n else 0.0,
            "cv_accuracy": None, "cv_std": None,
            "baseline_accuracy": 1.0 if n else 0.0, "lift": None,
            "cv_folds": 0, "n_samples": n, "classes": classes,
        }

    counts = Counter(y)
    baseline = max(counts.values()) / n  # always-guess-majority accuracy

    def _tree():
        return DecisionTreeClassifier(max_depth=max_depth, random_state=0)

    clf = _tree().fit(X, y)

    # Stratified CV needs >= 2 members per class; cap folds at the smallest class.
    folds = min(cv, min(counts.values()))
    if folds >= 2:
        scores = cross_val_score(
            _tree(), X, y, cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
        )
        cv_acc, cv_std = float(scores.mean()), float(scores.std())
        lift = cv_acc - baseline
    else:
        cv_acc = cv_std = lift = None
        folds = 0

    return {
        "rules": export_text(clf, feature_names=list(features.columns)),
        "train_accuracy": float(clf.score(X, y)),
        "cv_accuracy": cv_acc, "cv_std": cv_std,
        "baseline_accuracy": baseline, "lift": lift,
        "cv_folds": folds, "n_samples": n, "classes": classes,
    }


__all__ = [
    "winning_method",
    "win_rates",
    "method_advantage",
    "winner_feature_contrast",
    "divergence",
    "fit_winner_rule",
]
