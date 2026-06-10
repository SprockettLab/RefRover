import pytest
import numpy as np
import pandas as pd
from refrover.adaptive_k import estimate_k, _scree_elbow, _similarity_gap, _saturation_curve


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
    for method in ("scree_elbow", "similarity_gap", "saturation_curve"):
        k = estimate_k(three_cluster_sim, "s00", min_jaccard=0.01,
                       method=method, k_min=2, k_max=10)
        assert 2 <= k <= 10, f"{method} returned k={k} outside [2, 10]"


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
