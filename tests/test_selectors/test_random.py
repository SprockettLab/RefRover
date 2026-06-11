import pytest
from refrover.selectors import RandomSelector


def test_returns_k_results(clustered_sim):
    sel = RandomSelector(k=3, seed=0)
    result = sel.select(clustered_sim, "s00")
    assert len(result) == 3


def test_all_within_threshold(clustered_sim):
    sel = RandomSelector(k=4, min_jaccard=0.5, seed=1)
    result = sel.select(clustered_sim, "s00")
    for rid in result:
        assert clustered_sim.loc["s00", rid] >= 0.5


def test_fewer_than_k_candidates(sparse_sim):
    sel = RandomSelector(k=5, min_jaccard=0.1, seed=0)
    with pytest.warns(UserWarning, match="candidates available"):
        result = sel.select(sparse_sim, "s00")
    assert len(result) <= 5


def test_unknown_query_raises(sparse_sim):
    sel = RandomSelector(k=3, min_jaccard=0.1)
    with pytest.raises(KeyError, match="not found"):
        sel.select(sparse_sim, "does_not_exist")


def test_seed_reproducible(uniform_sim):
    sel = RandomSelector(k=4, seed=99)
    assert sel.select(uniform_sim, "s00") == sel.select(uniform_sim, "s00")


def test_results_are_valid_ids(clustered_sim):
    sel = RandomSelector(k=3, seed=0)
    result = sel.select(clustered_sim, "s00")
    assert all(rid in clustered_sim.columns for rid in result)
    assert len(result) == len(set(result))
