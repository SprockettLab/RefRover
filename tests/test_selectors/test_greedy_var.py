import pytest
from refrover.selectors import GreedyVarSelector


def test_returns_k_results(clustered_sim):
    sel = GreedyVarSelector(k=4)
    result = sel.select(clustered_sim, "s00")
    assert len(result) == 4


def test_query_always_first(clustered_sim):
    sel = GreedyVarSelector(k=4)
    result = sel.select(clustered_sim, "s00")
    assert result[0] == "s00"


def test_all_within_threshold(clustered_sim):
    sel = GreedyVarSelector(k=4, min_jaccard=0.05)
    result = sel.select(clustered_sim, "s00")
    for rid in result:
        assert clustered_sim.loc["s00", rid] >= 0.05


def test_spans_clusters(clustered_sim):
    """With min_jaccard low enough to allow cross-cluster candidates, diverse selection expected."""
    # cross-cluster sim = 0.05, so min_jaccard must be < 0.05 to allow them as candidates
    sel = GreedyVarSelector(k=4, min_jaccard=0.01)
    result = sel.select(clustered_sim, "s00")
    # cross-cluster assemblies have high variance in their similarity profiles
    # (0.7 within their own cluster, 0.05 to others) → GreedyVarSelector should prefer them
    cluster_a = {"s00", "s01", "s02", "s03"}
    from_other_clusters = [r for r in result if r not in cluster_a]
    assert len(from_other_clusters) >= 1, (
        "GreedyVarSelector should pick diverse prototypes spanning multiple clusters"
    )


def test_fewer_than_k_candidates_warns(sparse_sim):
    sel = GreedyVarSelector(k=5, min_jaccard=0.1)
    with pytest.warns(UserWarning, match="candidates available"):
        result = sel.select(sparse_sim, "s00")
    assert len(result) <= 5


def test_results_are_unique(clustered_sim):
    sel = GreedyVarSelector(k=5)
    result = sel.select(clustered_sim, "s00")
    assert len(result) == len(set(result))
