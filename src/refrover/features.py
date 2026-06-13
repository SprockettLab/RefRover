"""
Per-sample feature computation (PLAN.md §7.1, build step 4).

The primary deliverable of the whole project (§7) is *which cheaply-measurable
property of a sample predicts which selection method is best for it.* These are
the **predictors** for that model: a scalar feature vector per sample that the
§7.2 analysis regresses `winning_method ~ features` against.

Features fall into two groups by what they depend on:

**Sample-intrinsic** (from the sample's taxa-abundance profile — a property of the
microbial community, independent of which feature space we select on):
  - richness, shannon, pielou_evenness, simpson_dominance, berger_parker
  - species_distinctness — how independent the present taxa's dynamics are

**Feature-space-dependent** (from the uniform `samples × assemblies` matrix M of
the feature space in play):
  - n_candidates — references passing the mapping threshold for this sample
  - candidate_separability — how diverse the eligible references are
  - mean_candidate_signal — how strongly the sample maps to its candidates
  - saturation-curve shape (sat_auc, sat_k_to_90, sat_early_decay) — how fast the
    marginal differential signal decays as k grows (cf.
    `data/mouse_rewilding/saturation_curves.tsv`)

The saturation curve reuses the CSS greedy + frac_variance engine (CSS is
nested-consistent, so the first-k greedy picks *are* the curve), normalized to
total variance — same shape as the prototype, scale-free.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from refrover.rules import CSSRule, _correlation_distance
from refrover.scores import frac_variance


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _mean_pairwise_distance(columns: np.ndarray) -> float:
    """
    Mean off-diagonal correlation-distance among a set of column vectors.

    `columns` is observations × m. Returns the average 1 − Pearson over all m(m−1)
    ordered pairs (0 for fewer than two columns — separability is undefined).
    """
    m = columns.shape[1]
    if m < 2:
        return 0.0
    dist = _correlation_distance(columns)
    return float(dist.sum() / (m * (m - 1)))


# ──────────────────────────────────────────────────────────────────────────────
# Sample-intrinsic features (taxa-abundance profile)
# ──────────────────────────────────────────────────────────────────────────────
def diversity_features(abundance: pd.Series, *, min_abundance: float = 0.0) -> dict[str, float]:
    """
    Alpha-diversity descriptors of one sample's taxa-abundance vector (§7.1
    "diversity & evenness — one bug or fifty?").

    Returns richness, shannon (natural log), pielou_evenness, simpson_dominance
    (Σ p²), and berger_parker (max p). Abundances at or below `min_abundance` are
    treated as absent. Degenerate samples (0 or 1 present taxon) get evenness 0.
    """
    a = abundance.to_numpy(dtype=float)
    a = a[a > min_abundance]
    richness = int(a.size)
    if richness == 0:
        return {
            "richness": 0, "shannon": 0.0, "pielou_evenness": 0.0,
            "simpson_dominance": 0.0, "berger_parker": 0.0,
        }
    p = a / a.sum()
    shannon = float(-(p * np.log(p)).sum())
    evenness = float(shannon / np.log(richness)) if richness > 1 else 0.0
    return {
        "richness": richness,
        "shannon": shannon,
        "pielou_evenness": evenness,
        "simpson_dominance": float((p**2).sum()),
        "berger_parker": float(p.max()),
    }


def species_distinctness(
    abundance_matrix: pd.DataFrame, query_id: str, *, min_abundance: float = 0.0
) -> float:
    """
    Mean pairwise distance among the taxa present in `query_id` (§7.1 "species
    distinctness / separability — how distinct the species are").

    Each present taxon is embedded by its abundance profile across the cohort
    (its column in `abundance_matrix`); distinctness is the mean 1 − Pearson over
    those taxa. Taxa whose abundances co-vary tightly across samples are *less*
    distinct (redundant differential signal); independent dynamics are *more*
    distinct — the property that makes genomes separable by a binner. Returns 0.0
    when fewer than two taxa are present.
    """
    row = abundance_matrix.loc[query_id]
    present = row[row > min_abundance].index
    if len(present) < 2:
        return 0.0
    return _mean_pairwise_distance(abundance_matrix[present].to_numpy(dtype=float))


# ──────────────────────────────────────────────────────────────────────────────
# Feature-space-dependent features (the uniform samples × assemblies matrix)
# ──────────────────────────────────────────────────────────────────────────────
def candidate_features(
    M: pd.DataFrame, query_id: str, *, threshold: float = 0.0
) -> dict[str, float]:
    """
    Features of the *eligible reference set* for a sample under feature matrix M.

    - n_candidates: references whose signal to the query clears `threshold`
      (the mapping-threshold count from §7.1).
    - candidate_separability: mean pairwise correlation-distance among the
      candidate column vectors — how much distinct signal there is to choose from.
    - mean_candidate_signal: mean of the query's row over its candidates — how
      strongly the sample's reads map to eligible references.
    """
    row = M.loc[query_id]
    candidates = row[row >= threshold].index.tolist()
    n = len(candidates)
    return {
        "n_candidates": n,
        "candidate_separability": _mean_pairwise_distance(
            M[candidates].to_numpy(dtype=float)
        ),
        "mean_candidate_signal": float(row[candidates].mean()) if n else 0.0,
    }


def saturation_curve(
    M: pd.DataFrame, query_id: str, *, threshold: float = 0.0, k_max: int = 20
) -> pd.DataFrame:
    """
    Greedy marginal/cumulative differential-variance curve for a sample.

    Selects references greedily by CSS (the rule that maximises orthogonal-
    projection variance) and reports, at each step k, the cumulative fraction of
    total variance explained and the marginal gain over k−1. Because CSS is
    nested-consistent, the first-k picks are exactly the prefix of the full pick,
    so one pass yields the whole curve. Same shape as the
    `saturation_curves.tsv` prototype, here normalized to total variance.

    Returns a DataFrame with columns [k, marginal_var, cumulative_var].
    """
    n_candidates = int((M.loc[query_id] >= threshold).sum())
    k = min(k_max, n_candidates)
    picks = CSSRule(k=k, threshold=threshold).select(M, query_id)

    rows, prev = [], 0.0
    for kk in range(1, len(picks) + 1):
        cum = frac_variance(M, picks[:kk])
        rows.append((kk, cum - prev, cum))
        prev = cum
    return pd.DataFrame(rows, columns=["k", "marginal_var", "cumulative_var"])


def saturation_features(curve: pd.DataFrame) -> dict[str, float]:
    """
    Shape descriptors of a saturation curve (§7.1 "saturation-curve shape").

    - sat_auc: area under the cumulative curve normalized so its final value is 1,
      averaged over k. High → front-loaded: most signal bought by the first few
      references (selection barely matters past small k).
    - sat_k_to_90: smallest k reaching 90% of the final cumulative variance — an
      elbow proxy (low → saturates fast).
    - sat_early_decay: marginal gain at k=2 relative to k=1 (0 → the second pick
      adds almost nothing; ~1 → still climbing).
    """
    cum = curve["cumulative_var"].to_numpy(dtype=float)
    marg = curve["marginal_var"].to_numpy(dtype=float)
    n = cum.size
    if n == 0 or cum[-1] <= 0.0:
        return {"sat_auc": 0.0, "sat_k_to_90": 0.0, "sat_early_decay": 0.0}

    norm = cum / cum[-1]
    k_to_90 = int(np.argmax(norm >= 0.9)) + 1  # argmax finds first True
    early_decay = float(marg[1] / marg[0]) if n > 1 and marg[0] > 0 else 0.0
    return {
        "sat_auc": float(norm.mean()),
        "sat_k_to_90": float(k_to_90),
        "sat_early_decay": early_decay,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Assembly into a per-sample feature table
# ──────────────────────────────────────────────────────────────────────────────
def per_sample_features(
    M: pd.DataFrame,
    query_id: str,
    *,
    abundance: pd.DataFrame | None = None,
    threshold: float = 0.0,
    k_max: int = 20,
    min_abundance: float = 0.0,
) -> dict[str, float]:
    """
    Full per-sample feature vector for one sample (§7.1).

    Always computes the feature-space-dependent features from `M` (candidate set
    + saturation shape). If `abundance` (samples × taxa) is given, also computes
    the sample-intrinsic diversity + distinctness features from this sample's row;
    pass it once and it is reused across feature spaces.
    """
    feats: dict[str, float] = {}
    feats.update(candidate_features(M, query_id, threshold=threshold))
    feats.update(
        saturation_features(
            saturation_curve(M, query_id, threshold=threshold, k_max=k_max)
        )
    )
    if abundance is not None and query_id in abundance.index:
        feats.update(diversity_features(abundance.loc[query_id], min_abundance=min_abundance))
        feats["species_distinctness"] = species_distinctness(
            abundance, query_id, min_abundance=min_abundance
        )
    return feats


def feature_table(
    M: pd.DataFrame,
    *,
    abundance: pd.DataFrame | None = None,
    threshold: float = 0.0,
    k_max: int = 20,
    min_abundance: float = 0.0,
    query_ids: list[str] | None = None,
) -> pd.DataFrame:
    """
    Per-sample feature table: rows = samples, columns = features (§7.1).

    Computed for every row of `M` (or the given `query_ids`). This is the
    predictor matrix the §7.2 feature→method-choice model consumes.
    """
    ids = query_ids if query_ids is not None else list(M.index)
    records = {
        q: per_sample_features(
            M, q, abundance=abundance, threshold=threshold,
            k_max=k_max, min_abundance=min_abundance,
        )
        for q in ids
    }
    table = pd.DataFrame.from_dict(records, orient="index")
    table.index.name = "sample_id"
    return table


__all__ = [
    "diversity_features",
    "species_distinctness",
    "candidate_features",
    "saturation_curve",
    "saturation_features",
    "per_sample_features",
    "feature_table",
]
