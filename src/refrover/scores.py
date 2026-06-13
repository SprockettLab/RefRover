"""
Upstream score family (PLAN.md §6, build step 3).

True MAG quality only exists downstream (binning + CheckM2). Upstream, we build
proxies *shaped like* MAG yield and later test whether they predict the real
thing (the Tier-1-predicts-Tier-2 question). The unifying intuition (§6.2):

    "How many MAGs can this reference set yield" ≈ "how many independent,
     high-energy axes of cross-sample variation the selected references buy you."

That is computable from the singular-value spectrum of the selected columns. We
deliberately keep **three** proxies — which one best predicts MAG yield is itself
an open research question (§6.3) — all consuming the uniform `samples × assemblies`
feature matrix and a list of selected reference ids:

1. ``frac_variance``      — orthogonal-projection variance explained, in [0, 1]
                            (the generalized ``benchmark.unique_variance_explained``).
2. ``effective_rank``     — participation ratio of the selected columns' spectrum:
                            "how many independent axes, softly counted" (in [0, k]).
3. ``tiered_axis_count``  — *(default / most MAG-shaped)* weighted count of
                            resolvable axes: strong/clean axes (high tier) are paid
                            more than moderate ones (medium tier); noise is ignored.

All three ask "how much *non-redundant* differential signal did these references
buy?" — ``frac_variance`` blends it into one percentage, ``effective_rank`` counts
distinct signals, ``tiered_axis_count`` counts them but pays more for strong ones.

Per-sample aggregation is option (a) (§6.4): score each sample's own pick and keep
the distribution; the grid runner averages for a headline number. These functions
score a single selection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _selected_spectrum(matrix: pd.DataFrame, selected_ids: list[str]) -> np.ndarray:
    """
    Eigenvalues (variance along each principal axis) of the selected columns.

    The selected reference columns form a samples × k block; we mean-centre each
    column over samples and return the squared singular values (= covariance
    eigenvalues), sorted descending. These are the per-axis energies §6.2 tiers.
    Returns an empty array if nothing is selected or all energy is zero.

    Computed from the small k×k Gram matrix (Xᵀ X) via ``eigvalsh`` rather than an
    SVD of the tall samples×k block: the eigenvalues of Xᵀ X *are* the squared
    singular values, and the symmetric eigensolver is both faster and avoids the
    LAPACK SVD non-convergence that real feature matrices can trigger.
    """
    cols = [c for c in selected_ids if c in matrix.columns]
    if not cols:
        return np.empty(0)
    X = matrix[cols].to_numpy(dtype=float)
    X = np.nan_to_num(X - X.mean(axis=0, keepdims=True), nan=0.0, posinf=0.0, neginf=0.0)
    lam = np.linalg.eigvalsh(X.T @ X)          # ascending; tiny negatives are roundoff
    return np.sort(lam[lam > 0])[::-1]


def frac_variance(matrix: pd.DataFrame, selected_ids: list[str]) -> float:
    """
    Fraction of total cross-sample column variance explained by the selected
    columns via orthogonal projection (PLAN §6.3 metric 1).

    The full matrix (samples × references) is mean-centred; the selected columns
    span a subspace; we project all columns onto it and report
    ``(total_var − residual_var) / total_var``. Orthogonal projection means
    references correlated with the selected set count once, not multiple times —
    rewarding picks that span independent axes rather than piling onto one.

    Returns a value in [0, 1] (1.0 when the selected columns span the column
    space, e.g. all columns selected). This is the canonical implementation of
    the former ``benchmark.unique_variance_explained``.
    """
    cols = list(matrix.columns)
    pos = {c: i for i, c in enumerate(cols)}
    idx = [pos[s] for s in selected_ids if s in pos]

    X = matrix.to_numpy(dtype=float)
    X = np.nan_to_num(X - X.mean(axis=0), nan=0.0, posinf=0.0, neginf=0.0)
    total_var = float(np.var(X, axis=0, ddof=1).sum())
    if total_var == 0.0 or not idx:
        return 0.0

    Q, _ = np.linalg.qr(X[:, idx])  # orthonormal basis for the selected subspace
    resid = X - Q @ (Q.T @ X)
    resid_var = float(np.var(resid, axis=0, ddof=1).sum())
    return (total_var - resid_var) / total_var


def effective_rank(matrix: pd.DataFrame, selected_ids: list[str]) -> float:
    """
    Participation ratio of the selected columns' spectrum (PLAN §6.3 metric 2).

    PR = (Σ λ_i)² / Σ λ_i², where λ_i are the selected block's covariance
    eigenvalues. "How many independent axes, softly counted": 1.0 when the
    selection is effectively rank-1 (all references redundant), up to the number
    of selected references when they are mutually orthogonal and equal-energy.

    Returns 0.0 for an empty/zero-energy selection.
    """
    lam = _selected_spectrum(matrix, selected_ids)
    if lam.size == 0:
        return 0.0
    s = float(lam.sum())
    return (s * s) / float((lam**2).sum())


def tiered_axis_count(
    matrix: pd.DataFrame,
    selected_ids: list[str],
    *,
    high_ratio: float = 0.5,
    medium_ratio: float = 0.1,
    high_weight: float = 2.0,
    medium_weight: float = 1.0,
) -> float:
    """
    Weighted count of resolvable axes (PLAN §6.2/§6.3 metric 3 — the default,
    most MAG-shaped proxy).

    Each principal axis of the selected columns carries energy λ_i. We score axes
    by strength *relative to the dominant axis* (r_i = λ_i / λ_max) — scale-free,
    so k equal-energy independent axes all count, not just a fixed share of total:

        r_i >= high_ratio                    → "high" tier  (weight high_weight)
        medium_ratio <= r_i < high_ratio     → "medium" tier (weight medium_weight)
        r_i < medium_ratio                   → noise floor   (ignored)

    score = high_weight · n_high + medium_weight · n_medium.

    This matches "N high/medium MAGs, high worth more" far better than a single
    fraction. The thresholds/weights are heuristics to be calibrated against
    Tier-2 MAG yield; they are exposed for that tuning.

    Returns 0.0 for an empty/zero-energy selection.
    """
    lam = _selected_spectrum(matrix, selected_ids)
    if lam.size == 0:
        return 0.0
    r = lam / lam.max()
    n_high = int((r >= high_ratio).sum())
    n_medium = int(((r >= medium_ratio) & (r < high_ratio)).sum())
    return high_weight * n_high + medium_weight * n_medium


#: Name → scoring function, for the grid runner (build step 5).
SCORE_FUNCTIONS = {
    "frac_variance": frac_variance,
    "effective_rank": effective_rank,
    "tiered_axis_count": tiered_axis_count,
}


def score_selection(
    matrix: pd.DataFrame, selected_ids: list[str], **kwargs
) -> dict[str, float]:
    """
    Compute the whole score family for one selection.

    Returns ``{"frac_variance": ..., "effective_rank": ..., "tiered_axis_count": ...}``.
    Extra keyword args (e.g. ``high_ratio=``) are forwarded to the functions that
    accept them; functions that don't simply ignore them.
    """
    out: dict[str, float] = {}
    for name, fn in SCORE_FUNCTIONS.items():
        params = fn.__kwdefaults__ or {}
        accepted = {k: v for k, v in kwargs.items() if k in params}
        out[name] = fn(matrix, selected_ids, **accepted)
    return out


__all__ = [
    "frac_variance",
    "effective_rank",
    "tiered_axis_count",
    "score_selection",
    "SCORE_FUNCTIONS",
]
