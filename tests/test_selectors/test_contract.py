"""
Shared contract tests applied to every selector in SELECTOR_REGISTRY.

These lock the invariants the pipeline relies on, so a new selector that
violates the contract fails here rather than producing subtly wrong assignments.
"""

import pytest

from refrover.selectors import SELECTOR_REGISTRY

# Selectors runnable on a square similarity/containment matrix without extra deps.
# feedback is an unimplemented stub; archetype needs the optional `archetypes`
# package and is exercised in its own test module.
_CONTRACT_SELECTORS = ["random", "maxmin", "kmedoids", "greedy_var", "containment"]


def _make(name, **kw):
    return SELECTOR_REGISTRY[name](**kw)


@pytest.mark.parametrize("name", _CONTRACT_SELECTORS)
def test_returns_at_most_k(name, clustered_sim):
    sel = _make(name, k=3, min_jaccard=0.1)
    result = sel.select(clustered_sim, "s00")
    assert len(result) <= 3


@pytest.mark.parametrize("name", _CONTRACT_SELECTORS)
def test_results_unique(name, clustered_sim):
    sel = _make(name, k=4, min_jaccard=0.1)
    result = sel.select(clustered_sim, "s00")
    assert len(result) == len(set(result))


@pytest.mark.parametrize("name", _CONTRACT_SELECTORS)
def test_results_are_valid_candidates(name, clustered_sim):
    """Every returned id clears the threshold for the query."""
    sel = _make(name, k=3, min_jaccard=0.1)
    result = sel.select(clustered_sim, "s00")
    for rid in result:
        assert rid in clustered_sim.columns
        assert clustered_sim.loc["s00", rid] >= 0.1


@pytest.mark.parametrize("name", _CONTRACT_SELECTORS)
def test_query_included_first(name, clustered_sim):
    """The query's own assembly is always the first prototype."""
    sel = _make(name, k=3, min_jaccard=0.1)
    result = sel.select(clustered_sim, "s00")
    assert result[0] == "s00"


@pytest.mark.parametrize("name", _CONTRACT_SELECTORS)
def test_unknown_query_raises(name, clustered_sim):
    sel = _make(name, k=3, min_jaccard=0.1)
    with pytest.raises(KeyError):
        sel.select(clustered_sim, "not_a_sample")


@pytest.mark.parametrize("name", _CONTRACT_SELECTORS)
def test_fewer_candidates_than_k_warns(name, sparse_sim):
    sel = _make(name, k=6, min_jaccard=0.1)
    with pytest.warns(UserWarning):
        result = sel.select(sparse_sim, "s00")
    assert len(result) <= 6
