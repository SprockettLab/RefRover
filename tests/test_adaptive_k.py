import pytest
import numpy as np
import pandas as pd
from refrover.adaptive_k import (
    estimate_k, _scree_elbow, _similarity_gap, _saturation_curve,
    _containment_saturation,
)


@pytest.fixture
def three_cluster_sim():
    """
    15 samples, 3 tight clusters of 5.
    Within-cluster Jaccard ~ 0.7, cross-cluster ~ 0.05.
    The intrinsic dimensionality is ~3 → expect k_hat around 3.
    """
    rng = np.random.default_rng(0)
    n = 15
    ids = [f"s{i:02d}" for i in range(n)]
    m = np.full((n, n), 0.05)
    for start in (0, 5, 10):
        for i in range(start, start + 5):
            for j in range(start, start + 5):
                if i != j:
                    m[i, j] = 0.65 + 0.1 * rng.random()
    np.fill_diagonal(m, 1.0)
    sym = (m + m.T) / 2
    np.fill_diagonal(sym, 1.0)
    return pd.DataFrame(sym, index=ids, columns=ids)


def test_estimate_k_returns_int(three_cluster_sim):
    k = estimate_k(three_cluster_sim, "s00", min_jaccard=0.01)
    assert isinstance(k, int)


def test_estimate_k_respects_k_min(three_cluster_sim):
    k = estimate_k(three_cluster_sim, "s00", min_jaccard=0.01, k_min=4)
    assert k >= 4


def test_estimate_k_respects_k_max(three_cluster_sim):
    k = estimate_k(three_cluster_sim, "s00", min_jaccard=0.01, k_max=5)
    assert k <= 5


def test_estimate_k_unknown_query(three_cluster_sim):
    with pytest.raises(KeyError):
        estimate_k(three_cluster_sim, "does_not_exist")


def test_all_methods_return_valid_k(three_cluster_sim):
    for method in ("scree_elbow", "similarity_gap", "saturation_curve",
                   "containment_saturation"):
        k = estimate_k(three_cluster_sim, "s00", min_jaccard=0.01,
                       method=method, k_min=2, k_max=10)
        assert 2 <= k <= 10, f"{method} returned k={k} outside [2, 10]"


def test_containment_saturation_finds_low_dim(three_cluster_sim):
    """3-cluster data: the greedy residual-variance elbow should be small."""
    ids = three_cluster_sim.columns.tolist()
    k = _containment_saturation(three_cluster_sim, ids, k_min=1, k_max=12)
    assert 1 <= k <= 6


def test_containment_saturation_monotone_curve_threshold():
    """
    A matrix with one dominant variance axis and a long tail of tiny ones
    should saturate after very few prototypes.
    """
    rng = np.random.default_rng(1)
    n = 20
    base = rng.normal(size=n)
    cols = {"big0": base * 5.0, "big1": rng.normal(size=n) * 4.0}
    for i in range(8):
        cols[f"tiny{i}"] = base * 5.0 + 1e-4 * rng.normal(size=n)  # near-duplicates
    df = pd.DataFrame(cols, index=[f"s{i}" for i in range(n)])
    k = _containment_saturation(df, list(df.columns), k_min=1, k_max=10)
    assert k <= 3  # only ~2 real axes of variation


def test_similarity_gap_finds_natural_break():
    """
    Construct a similarity row with a clear gap: 3 high-sim candidates,
    then a large drop, then low-sim candidates.
    similarity_gap should return k=3.
    """
    ids = [f"s{i}" for i in range(8)]
    sims = [1.0, 0.75, 0.70, 0.68,   # top 4 — tight group
            0.15, 0.12, 0.11, 0.10]   # bottom 4 — after big drop
    row = pd.Series(dict(zip(ids, sims)))
    candidates = ids

    k = _similarity_gap(row, candidates, k_min=2, k_max=8)
    assert k == 4  # gap is between index 3 (0.68) and 4 (0.15)


def test_saturation_curve_stops_at_elbow(three_cluster_sim):
    """With 3 clusters, saturation should stop adding prototypes at ~3."""
    ids = three_cluster_sim.columns.tolist()
    k = _saturation_curve(three_cluster_sim, "s00", ids, k_min=2, k_max=10)
    # After the first 3 prototypes (one per cluster) diversity gain collapses
    assert k <= 6  # generous upper bound


def test_scree_elbow_clustered(three_cluster_sim):
    """3-cluster data has intrinsic dim ~3; scree elbow should be near that."""
    ids = three_cluster_sim.columns.tolist()
    k = _scree_elbow(three_cluster_sim, ids, k_min=2, k_max=10)
    assert 2 <= k <= 6
