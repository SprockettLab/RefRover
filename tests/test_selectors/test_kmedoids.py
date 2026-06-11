from refrover.selectors import KMedoidsSelector


def test_returns_k_results(clustered_sim):
    sel = KMedoidsSelector(k=3)
    assert len(sel.select(clustered_sim, "s00")) == 3


def test_query_always_included(clustered_sim):
    sel = KMedoidsSelector(k=3, min_jaccard=0.0)
    result = sel.select(clustered_sim, "s04")
    assert "s04" in result


def test_medoids_are_valid_ids(clustered_sim):
    sel = KMedoidsSelector(k=4, min_jaccard=0.0)
    result = sel.select(clustered_sim, "s00")
    assert all(r in clustered_sim.columns for r in result)
    assert len(result) == len(set(result))


def test_reproduces_with_same_seed(clustered_sim):
    sel = KMedoidsSelector(k=3, random_state=7)
    assert sel.select(clustered_sim, "s00") == sel.select(clustered_sim, "s00")


def test_covers_clusters(clustered_sim):
    """k-medoids should select one representative from each cluster."""
    sel = KMedoidsSelector(k=3, min_jaccard=0.0)
    result = sel.select(clustered_sim, "s00")
    assert len({int(r[1:]) // 4 for r in result}) == 3
