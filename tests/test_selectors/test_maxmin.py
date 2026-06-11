import numpy as np
import pandas as pd
import pytest
from refrover.selectors import MaxMinSelector


def test_query_is_first(clustered_sim):
    sel = MaxMinSelector(k=3)
    result = sel.select(clustered_sim, "s00")
    assert result[0] == "s00"


def test_returns_k_results(clustered_sim):
    sel = MaxMinSelector(k=4)
    assert len(sel.select(clustered_sim, "s00")) == 4


def test_selects_across_clusters(clustered_sim):
    """With k=3 and 3 clusters, MaxMin should pick one from each cluster."""
    sel = MaxMinSelector(k=3, min_jaccard=0.0)
    result = sel.select(clustered_sim, "s00")
    cluster_of = lambda sid: int(sid[1:]) // 4
    clusters_hit = {cluster_of(r) for r in result}
    assert len(clusters_hit) == 3


def test_diversity_exceeds_random(uniform_sim):
    """MaxMin selection should be more diverse than a random selection."""
    sel_mm = MaxMinSelector(k=5, min_jaccard=0.0)
    from refrover.selectors import RandomSelector
    sel_r = RandomSelector(k=5, min_jaccard=0.0, seed=0)

    def min_pairwise_dist(ids, sim):
        vals = [1 - sim.loc[a, b] for i, a in enumerate(ids) for b in ids[i+1:]]
        return min(vals) if vals else 0.0

    mm_div = min_pairwise_dist(sel_mm.select(uniform_sim, "s00"), uniform_sim)
    r_div = min_pairwise_dist(sel_r.select(uniform_sim, "s00"), uniform_sim)
    assert mm_div >= r_div


def test_all_within_threshold(clustered_sim):
    sel = MaxMinSelector(k=3, min_jaccard=0.5)
    result = sel.select(clustered_sim, "s00")
    for rid in result:
        assert clustered_sim.loc["s00", rid] >= 0.5


def test_no_duplicates(clustered_sim):
    sel = MaxMinSelector(k=4)
    result = sel.select(clustered_sim, "s00")
    assert len(result) == len(set(result))
