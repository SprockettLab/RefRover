"""
Adaptive k selection: determine the number of prototypes per sample
from the dataset's similarity structure rather than using a fixed global k.

The core idea: as k increases, each additional prototype adds less new
differential signal (diminishing returns). The optimal k is at the elbow
of the signal-saturation curve.

Three estimators are implemented:

scree_elbow
    PCoA on the Jaccard distance matrix; fit an exponential decay to the
    eigenvalue spectrum; elbow = point where marginal explained variance
    drops below a threshold. Analogous to scree plot in PCA.

similarity_gap
    Sort candidate similarities to the query in descending order.
    Find the largest gap in the sorted similarity values — the natural
    break between "similar enough to add signal" and "too distant to matter".
    Simple, fast, interpretable.

saturation_curve
    For increasing k, estimate the marginal diversity gain from adding
    the k-th prototype (using MaxMin distance). Stop when the gain drops
    below a fraction of the initial gain.

All return an integer k_hat. The CLI exposes --k auto to trigger this.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Literal


AdaptiveMethod = Literal["scree_elbow", "similarity_gap", "saturation_curve"]


def estimate_k(
    sim_matrix: pd.DataFrame,
    query_id: str,
    min_jaccard: float = 0.1,
    method: AdaptiveMethod = "similarity_gap",
    k_min: int = 3,
    k_max: int = 20,
) -> int:
    """
    Estimate the optimal number of prototypes for query_id.

    Parameters
    ----------
    sim_matrix : square Jaccard similarity DataFrame
    query_id : row/column label for the query sample
    min_jaccard : minimum similarity threshold (same as selector)
    method : which estimator to use
    k_min, k_max : bounds on returned k

    Returns
    -------
    int : estimated k, clipped to [k_min, k_max]
    """
    if query_id not in sim_matrix.index:
        raise KeyError(f"query_id '{query_id}' not found in similarity matrix")

    # Restrict to valid candidates
    row = sim_matrix.loc[query_id]
    candidates = row[row >= min_jaccard].index.tolist()
    n_candidates = len(candidates)

    if n_candidates <= k_min:
        return n_candidates

    if method == "scree_elbow":
        k_hat = _scree_elbow(sim_matrix, candidates, k_min, k_max)
    elif method == "similarity_gap":
        k_hat = _similarity_gap(row, candidates, k_min, k_max)
    elif method == "saturation_curve":
        k_hat = _saturation_curve(sim_matrix, query_id, candidates, k_min, k_max)
    else:
        raise ValueError(f"Unknown adaptive k method: {method!r}")

    return int(np.clip(k_hat, k_min, min(k_max, n_candidates)))


# ── Estimators ────────────────────────────────────────────────────────────────

def _scree_elbow(
    sim_matrix: pd.DataFrame,
    candidates: list[str],
    k_min: int,
    k_max: int,
) -> int:
    """
    PCoA scree elbow: eigenvalue-based estimate of intrinsic dimensionality.

    Converts Jaccard similarity to distance, double-centres, takes eigenvalues.
    The elbow in the sorted positive eigenvalue spectrum indicates the number
    of meaningful axes of variation — a natural estimate for k.
    """
    sub = sim_matrix.loc[candidates, candidates]
    dist = 1.0 - sub.values.astype(float)
    np.fill_diagonal(dist, 0.0)

    # Double-centring (classical MDS)
    n = len(candidates)
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ (dist ** 2) @ H

    eigvals = np.linalg.eigvalsh(B)
    eigvals = np.sort(eigvals)[::-1]
    pos = eigvals[eigvals > 0]

    if len(pos) < 2:
        return k_min

    # Kneedle-style: find index with maximum curvature in normalised scree plot
    x = np.arange(len(pos), dtype=float)
    y = pos / pos.sum()
    # Straight line from first to last point
    line_y = y[0] + (y[-1] - y[0]) * x / x[-1]
    deviations = y - line_y
    elbow_idx = int(np.argmax(deviations)) + 1  # 1-indexed k

    return max(k_min, elbow_idx)


def _similarity_gap(
    query_row: pd.Series,
    candidates: list[str],
    k_min: int,
    k_max: int,
) -> int:
    """
    Largest-gap heuristic: sort candidate similarities descending, find the
    biggest drop. Prototypes above the gap add meaningful signal; below it
    are barely different from the background.
    """
    sims = query_row[candidates].sort_values(ascending=False).values
    if len(sims) < 2:
        return k_min

    gaps = np.diff(sims)           # all negative (descending)
    gap_idx = int(np.argmin(gaps)) # index of the biggest drop
    k_hat = gap_idx + 1            # number of prototypes above the gap

    return max(k_min, k_hat)


def _saturation_curve(
    sim_matrix: pd.DataFrame,
    query_id: str,
    candidates: list[str],
    k_min: int,
    k_max: int,
) -> int:
    """
    Greedy diversity saturation: incrementally add the MaxMin-greedy prototype
    and track the marginal gain in minimum pairwise distance. The elbow is the
    largest consecutive drop in the gains sequence — the point where adding
    another prototype stops providing meaningfully new diversity.
    """
    if len(candidates) == 0:
        return k_min

    sub = sim_matrix.loc[candidates, candidates]
    dist = 1.0 - sub.values.astype(float)
    np.fill_diagonal(dist, 0.0)
    idx = {c: i for i, c in enumerate(candidates)}

    first = query_id if query_id in idx else candidates[0]
    selected = [idx[first]]
    remaining = [i for i in range(len(candidates)) if i != idx[first]]

    min_dist = dist[selected[0], remaining].copy()
    gains = []

    while remaining and len(selected) < k_max:
        best_local_idx = int(np.argmax(min_dist))
        gain = float(min_dist[best_local_idx])
        gains.append(gain)

        best = remaining[best_local_idx]
        selected.append(best)
        remaining.pop(best_local_idx)
        min_dist = np.delete(min_dist, best_local_idx)

        if remaining:
            min_dist = np.minimum(min_dist, dist[best, remaining])

    if len(gains) < 2:
        return max(k_min, len(gains) + 1)

    # Elbow = largest consecutive drop in gains
    drops = np.diff(gains)           # all non-positive (gains are non-increasing)
    elbow_idx = int(np.argmin(drops)) # index of biggest drop
    # +1 for seed, +1 because we want k prototypes *including* the seed
    k_hat = elbow_idx + 2

    return max(k_min, k_hat)
