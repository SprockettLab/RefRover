import pytest
from refrover.selectors import ArchetypeSelector


def test_returns_k_results(clustered_sim):
    sel = ArchetypeSelector(k=3)
    assert len(sel.select(clustered_sim, "s00")) == 3


def test_query_always_included(clustered_sim):
    sel = ArchetypeSelector(k=3, min_jaccard=0.0)
    assert "s08" in sel.select(clustered_sim, "s08")


def test_no_duplicates(clustered_sim):
    sel = ArchetypeSelector(k=4, min_jaccard=0.0)
    result = sel.select(clustered_sim, "s00")
    assert len(result) == len(set(result))


def test_spans_clusters(clustered_sim):
    """Archetypes should select samples from all 3 distinct clusters."""
    sel = ArchetypeSelector(k=3, min_jaccard=0.0)
    result = sel.select(clustered_sim, "s00")
    assert len({int(r[1:]) // 4 for r in result}) == 3


def test_valid_sample_ids(clustered_sim):
    sel = ArchetypeSelector(k=3)
    result = sel.select(clustered_sim, "s00")
    assert all(r in clustered_sim.columns for r in result)


def test_fewer_candidates_than_k(sparse_sim):
    sel = ArchetypeSelector(k=5, min_jaccard=0.1)
    with pytest.warns(UserWarning, match="candidates available"):
        result = sel.select(sparse_sim, "s00")
    assert len(result) <= 5
