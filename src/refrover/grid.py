"""
The grid runner (PLAN.md §9 step 5) — feature × rule × k → {scores} + overlap.

This is where the three orthogonal knobs of PLAN §1 meet:

    FEATURE SPACE   ×   SELECTION RULE   ×   k    →   reference set   →   SCORE
    (which matrix)      (which algorithm)   (budget)   (per sample)       (per sample)

For every (feature_space, rule, k, sample) the runner selects a reference set and
scores it with the whole §6 family, emitting the tidy per-sample table that §7's
feature→method-choice analysis consumes. It also emits the §7.3 **overlap table**:
per (feature_space, k, sample), the Jaccard of chosen reference-ID sets between
every pair of rules — "where methods agree the choice doesn't matter; where they
diverge, the score (and ultimately the per-sample features) arbitrate."

Each selection is computed exactly once; both tables derive from it. Per-sample
aggregation is option (a) (§6.4): we keep the full per-sample distribution and let
the caller average for a headline number.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from itertools import combinations

import pandas as pd

from refrover.rules import RULE_REGISTRY
from refrover.scores import score_selection


@dataclass
class FeatureSpaceSpec:
    """
    One feature space for the grid: a uniform `samples × assemblies` matrix plus
    its selection settings.

    Parameters
    ----------
    name : str
        Stable id used in the grid's `feature_space` column (e.g. "jaccard").
    matrix : pd.DataFrame
        The uniform matrix from `feature_spaces.py`.
    threshold : float
        Candidate filter for this feature space (min Jaccard / min containment /
        min abundance — they live in the same code, differ only in value).
    clades : pd.Series, optional
        reference_id -> clade label. Required for `taxonomy_stratified`; its
        absence marks that rule invalid on this feature space (PLAN §5 grid:
        taxonomy_stratified is ✗ for Jaccard/containment, ✓ for GTDB).
    """

    name: str
    matrix: pd.DataFrame
    threshold: float = 0.0
    clades: pd.Series | None = field(default=None, repr=False)


@dataclass
class GridResult:
    """Outputs of :func:`run_grid`."""

    scores: pd.DataFrame      # tidy: sample × feature_space × rule × k → score family
    overlap: pd.DataFrame     # tidy: feature_space × k × sample × (rule_a, rule_b) → jaccard
    selections: pd.DataFrame  # raw picks: feature_space × rule × k × sample → selected ids


def _rule_valid_for(rule_name: str, spec: FeatureSpaceSpec) -> bool:
    """PLAN §5 validity grid: taxonomy_stratified needs clade labels."""
    if rule_name == "taxonomy_stratified":
        return spec.clades is not None
    return True


def _instantiate(rule_name: str, k: int, spec: FeatureSpaceSpec):
    cls = RULE_REGISTRY[rule_name]
    if rule_name == "taxonomy_stratified":
        return cls(k=k, threshold=spec.threshold, clades=spec.clades)
    return cls(k=k, threshold=spec.threshold)


def _jaccard(a: set, b: set) -> float:
    """Jaccard of two reference-id sets (1.0 for two empty sets)."""
    union = a | b
    return 1.0 if not union else len(a & b) / len(union)


def run_grid(
    feature_spaces: list[FeatureSpaceSpec],
    rules: list[str],
    k_values: list[int],
    *,
    query_ids: list[str] | None = None,
    score_kwargs: dict | None = None,
) -> GridResult:
    """
    Run the full feature × rule × k grid.

    Parameters
    ----------
    feature_spaces : list[FeatureSpaceSpec]
        The feature spaces to select on (each carries its matrix + threshold).
    rules : list[str]
        Rule ids from ``rules.RULE_REGISTRY``. Combinations invalid for a feature
        space (per §5) are skipped. A rule needing an optional dependency that is
        not installed (e.g. ``archetype``) is skipped grid-wide with a warning.
    k_values : list[int]
        Budgets to sweep.
    query_ids : list[str], optional
        Samples to score (default: every row of each feature space's matrix,
        intersected with this list when given).
    score_kwargs : dict, optional
        Forwarded to the score family (e.g. ``high_ratio=`` for tiered_axis_count).

    Returns
    -------
    GridResult
        ``scores`` (tidy per-sample score family), ``overlap`` (pairwise rule
        agreement per sample/k), and ``selections`` (the raw picks).
    """
    score_kwargs = score_kwargs or {}
    score_records: list[dict] = []
    selection_records: list[dict] = []
    skip_rules: set[str] = set()  # rules disabled grid-wide (missing optional dep)

    for spec in feature_spaces:
        M = spec.matrix
        ids = list(M.index) if query_ids is None else [q for q in query_ids if q in M.index]

        for rule_name in rules:
            if rule_name in skip_rules or not _rule_valid_for(rule_name, spec):
                continue
            for k in k_values:
                rule = _instantiate(rule_name, k, spec)
                for q in ids:
                    try:
                        with warnings.catch_warnings():
                            # "fewer candidates than k" is expected; n_selected records it.
                            warnings.simplefilter("ignore", UserWarning)
                            selected = rule.select(M, q)
                    except ImportError as exc:
                        warnings.warn(
                            f"Rule '{rule_name}' unavailable ({exc}); skipping it.",
                            stacklevel=2,
                        )
                        skip_rules.add(rule_name)
                        break  # abandon this rule for all remaining samples
                    else:
                        scores = score_selection(M, selected, **score_kwargs)
                        base = {
                            "sample_id": q,
                            "feature_space": spec.name,
                            "rule": rule_name,
                            "k": k,
                            "n_selected": len(selected),
                        }
                        score_records.append({**base, **scores})
                        selection_records.append(
                            {**base, "selected": tuple(selected)}
                        )
                else:
                    continue  # only reached if the sample loop didn't break
                break  # rule hit ImportError -> stop sweeping k for it

    scores = pd.DataFrame(score_records)
    selections = pd.DataFrame(selection_records)
    overlap = _overlap_from_selections(selections)
    return GridResult(scores=scores, overlap=overlap, selections=selections)


def _overlap_from_selections(selections: pd.DataFrame) -> pd.DataFrame:
    """Pairwise rule-agreement (Jaccard of chosen ids) per feature_space/k/sample."""
    if selections.empty:
        return pd.DataFrame(
            columns=["feature_space", "k", "sample_id", "rule_a", "rule_b", "jaccard"]
        )

    rows: list[dict] = []
    grouped = selections.groupby(["feature_space", "k", "sample_id"], sort=False)
    for (fs, k, sample), block in grouped:
        picks = dict(zip(block["rule"], block["selected"]))
        for rule_a, rule_b in combinations(sorted(picks), 2):
            rows.append(
                {
                    "feature_space": fs,
                    "k": k,
                    "sample_id": sample,
                    "rule_a": rule_a,
                    "rule_b": rule_b,
                    "jaccard": _jaccard(set(picks[rule_a]), set(picks[rule_b])),
                }
            )
    return pd.DataFrame(rows)


def overlap_matrix(
    overlap: pd.DataFrame, feature_space: str, k: int
) -> pd.DataFrame:
    """
    Pivot the long overlap table into a symmetric rule × rule mean-Jaccard matrix
    for one (feature_space, k) — the layout of the
    ``selector_agreement_k5.tsv`` prototype. Diagonal is 1.0; the off-diagonal is
    the mean agreement across samples.
    """
    sub = overlap[(overlap["feature_space"] == feature_space) & (overlap["k"] == k)]
    rules = sorted(set(sub["rule_a"]) | set(sub["rule_b"]))
    mat = pd.DataFrame(1.0, index=rules, columns=rules)
    means = sub.groupby(["rule_a", "rule_b"])["jaccard"].mean()
    for (a, b), val in means.items():
        mat.loc[a, b] = val
        mat.loc[b, a] = val
    return mat


def headline(scores: pd.DataFrame) -> pd.DataFrame:
    """
    Mean score per (feature_space, rule, k) across samples (§6.4 headline number).

    Keeps the per-sample distribution in ``scores``; this is just the average for
    a quick leaderboard. Sorted by tiered_axis_count (the default proxy) desc.
    """
    metrics = ["frac_variance", "effective_rank", "tiered_axis_count"]
    agg = (
        scores.groupby(["feature_space", "rule", "k"])[metrics + ["n_selected"]]
        .mean()
        .reset_index()
    )
    return agg.sort_values("tiered_axis_count", ascending=False, ignore_index=True)


__all__ = [
    "FeatureSpaceSpec",
    "GridResult",
    "run_grid",
    "overlap_matrix",
    "headline",
]
