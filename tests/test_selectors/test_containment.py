import numpy as np
import pandas as pd
import pytest
from refrover.selectors.containment import ContainmentSelector
from refrover.selectors.base import BaseSelector
from refrover.selectors import SELECTOR_REGISTRY, CONTAINMENT_SELECTORS


@pytest.fixture
def containment_mat():
    """
    6 samples x 6 assemblies.
    Diagonal is 1.0 (each sample's own assembly).
    Samples in the same 'group' share similar containment profiles.
    Group A: s0, s1, s2 (assemblies a0-a2, contain each other's reads well)
    Group B: s3, s4, s5 (assemblies a3-a5)
    Cross-group containment is low (~0.05).
    """
    rng = np.random.default_rng(42)
    sample_ids = [f"s{i}" for i in range(6)]
    asm_ids    = [f"s{i}" for i in range(6)]  # assembly IDs match sample IDs

    mat = np.full((6, 6), 0.04)
    for grp in [(0, 1, 2), (3, 4, 5)]:
        for i in grp:
            for j in grp:
                if i == j:
                    mat[i, j] = 1.0
                else:
                    mat[i, j] = 0.3 + 0.2 * rng.random()

    return pd.DataFrame(mat, index=sample_ids, columns=asm_ids)


def test_returns_k_results(containment_mat):
    sel = ContainmentSelector(k=3, min_containment=0.05)
    result = sel.select(containment_mat, "s0")
    assert len(result) == 3


def test_own_assembly_first(containment_mat):
    sel = ContainmentSelector(k=3, min_containment=0.05)
    result = sel.select(containment_mat, "s0")
    assert result[0] == "s0"


def test_all_meet_min_containment(containment_mat):
    sel = ContainmentSelector(k=3, min_containment=0.1)
    result = sel.select(containment_mat, "s0")
    for rid in result:
        assert containment_mat.loc["s0", rid] >= 0.1


def test_spans_groups_when_threshold_low(containment_mat):
    """With min_containment=0.01 all assemblies are candidates; diversity should pull cross-group."""
    sel = ContainmentSelector(k=4, min_containment=0.01)
    result = sel.select(containment_mat, "s0")
    group_a = {"s0", "s1", "s2"}
    from_b = [r for r in result if r not in group_a]
    assert len(from_b) >= 1


def test_unknown_query_raises(containment_mat):
    sel = ContainmentSelector(k=3)
    with pytest.raises(KeyError, match="not found"):
        sel.select(containment_mat, "does_not_exist")


def test_fewer_than_k_warns(containment_mat):
    # min_containment=0.5 leaves only the diagonal (1.0) in range
    sel = ContainmentSelector(k=4, min_containment=0.5)
    with pytest.warns(UserWarning, match="candidates available"):
        result = sel.select(containment_mat, "s0")
    assert len(result) <= 4


def test_weighted_selection_favours_high_weight(containment_mat):
    """Assembly s3 normally wouldn't be selected (cross-group), but with a high weight it should be."""
    weights = pd.Series({"s3": 10.0}, dtype=float)  # strongly favour s3
    sel = ContainmentSelector(k=3, min_containment=0.01, weights=weights)
    result = sel.select(containment_mat, "s0")
    assert "s3" in result


def test_results_are_unique(containment_mat):
    sel = ContainmentSelector(k=4, min_containment=0.05)
    result = sel.select(containment_mat, "s0")
    assert len(result) == len(set(result))


# ── Registry / interface integration ──────────────────────────────────────────

def test_is_base_selector_subclass():
    assert issubclass(ContainmentSelector, BaseSelector)


def test_registered_in_registry():
    assert SELECTOR_REGISTRY["containment"] is ContainmentSelector
    assert "containment" in CONTAINMENT_SELECTORS


def test_constructible_via_registry_min_jaccard(containment_mat):
    """The registry constructs selectors uniformly with k= and min_jaccard=."""
    sel_cls = SELECTOR_REGISTRY["containment"]
    sel = sel_cls(k=3, min_jaccard=0.05)
    assert sel.min_containment == 0.05
    result = sel.select(containment_mat, "s0")
    assert result[0] == "s0"
    assert len(result) == 3


def test_min_jaccard_overrides_min_containment():
    sel = ContainmentSelector(k=3, min_containment=0.05, min_jaccard=0.2)
    assert sel.min_containment == 0.2


def test_base_validation_applies():
    """k < 1 should raise via BaseSelector validation."""
    with pytest.raises(ValueError, match="k must be"):
        ContainmentSelector(k=0)


def test_returns_only_threshold_candidates(containment_mat):
    """All returned prototypes clear the containment threshold for the query."""
    sel = ContainmentSelector(k=6, min_containment=0.05)
    result = sel.select(containment_mat, "s0")
    assert all(containment_mat.loc["s0", r] >= 0.05 for r in result)
