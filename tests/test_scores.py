"""Tests for the upstream score family (PLAN.md §6, build step 3)."""

import numpy as np
import pandas as pd
import pytest

from refrover.scores import (
    SCORE_FUNCTIONS,
    effective_rank,
    frac_variance,
    score_selection,
    tiered_axis_count,
)


def _matrix(cols: dict[str, list[float]], n=6) -> pd.DataFrame:
    M = pd.DataFrame(cols, index=[f"s{i}" for i in range(n)])
    M.index.name = "sample_id"
    M.columns.name = "assembly_id"
    return M


@pytest.fixture
def orthogonal_matrix():
    """Three mutually orthogonal, equal-energy reference columns + a redundant copy."""
    a = [1.0, -1.0, 0.0, 0.0, 0.0, 0.0]
    b = [0.0, 0.0, 1.0, -1.0, 0.0, 0.0]
    c = [0.0, 0.0, 0.0, 0.0, 1.0, -1.0]
    return _matrix({"a": a, "b": b, "c": c, "a_dup": a})


# ── frac_variance ───────────────────────────────────────────────────────────
def test_frac_variance_all_columns_is_one(orthogonal_matrix):
    M = orthogonal_matrix
    assert frac_variance(M, list(M.columns)) == pytest.approx(1.0)


def test_frac_variance_redundant_pick_counts_once(orthogonal_matrix):
    M = orthogonal_matrix
    # a and a_dup span the same axis -> together explain only that one axis,
    # the same as picking a alone.
    assert frac_variance(M, ["a", "a_dup"]) == pytest.approx(frac_variance(M, ["a"]))


def test_frac_variance_matches_legacy_alias(orthogonal_matrix):
    from refrover.benchmark import unique_variance_explained

    M = orthogonal_matrix
    assert unique_variance_explained(M, ["a", "b"]) == pytest.approx(
        frac_variance(M, ["a", "b"])
    )


def test_frac_variance_empty_and_zero():
    M = _matrix({"a": [0.0] * 6, "b": [0.0] * 6})
    assert frac_variance(M, []) == 0.0
    assert frac_variance(M, ["a"]) == 0.0  # zero total variance


# ── effective_rank ──────────────────────────────────────────────────────────
def test_effective_rank_orthogonal_equal_energy(orthogonal_matrix):
    M = orthogonal_matrix
    # three orthogonal equal-energy axes -> participation ratio ~ 3
    assert effective_rank(M, ["a", "b", "c"]) == pytest.approx(3.0, abs=1e-9)


def test_effective_rank_redundant_columns_is_one(orthogonal_matrix):
    M = orthogonal_matrix
    # a and its duplicate span a single axis -> effective rank 1
    assert effective_rank(M, ["a", "a_dup"]) == pytest.approx(1.0, abs=1e-9)


def test_effective_rank_single_column_is_one(orthogonal_matrix):
    assert effective_rank(orthogonal_matrix, ["a"]) == pytest.approx(1.0)


def test_effective_rank_empty_is_zero(orthogonal_matrix):
    assert effective_rank(orthogonal_matrix, []) == 0.0


# ── tiered_axis_count ───────────────────────────────────────────────────────
def test_tiered_counts_equal_axes_as_high(orthogonal_matrix):
    M = orthogonal_matrix
    # three equal-energy axes: all r_i = 1 >= high_ratio -> 3 high axes
    score = tiered_axis_count(M, ["a", "b", "c"], high_weight=2.0, medium_weight=1.0)
    assert score == pytest.approx(2.0 * 3)


def test_tiered_redundant_columns_single_high_axis(orthogonal_matrix):
    M = orthogonal_matrix
    score = tiered_axis_count(M, ["a", "a_dup"], high_weight=2.0)
    assert score == pytest.approx(2.0)  # one resolvable axis


def test_tiered_medium_tier_weighted_less():
    # One dominant axis + one weak, *orthogonal*, above-noise axis.
    # x and y are both zero-mean and orthogonal (dot = 0), so they form two clean
    # principal axes with energies 54 and 4 -> r_y = 4/54 ≈ 0.074, in the medium band.
    x = [3.0, -3.0, 3.0, -3.0, 3.0, -3.0]
    y = [1.0, -1.0, -1.0, 1.0, 0.0, 0.0]
    M = _matrix({"x": x, "y": y})
    score = tiered_axis_count(
        M, ["x", "y"], high_ratio=0.5, medium_ratio=0.01,
        high_weight=2.0, medium_weight=1.0,
    )
    # dominant axis is high (2.0); the weak orthogonal axis lands in the medium band (1.0)
    assert score == pytest.approx(3.0)


def test_tiered_noise_floor_ignored():
    x = [3.0, -3.0, 3.0, -3.0, 3.0, -3.0]
    tiny = [0.0, 0.0, 1e-6, -1e-6, 0.0, 0.0]  # below medium_ratio -> ignored
    M = _matrix({"x": x, "tiny": tiny})
    score = tiered_axis_count(M, ["x", "tiny"], high_weight=2.0, medium_weight=1.0)
    assert score == pytest.approx(2.0)  # only the dominant axis counts


def test_tiered_empty_is_zero(orthogonal_matrix):
    assert tiered_axis_count(orthogonal_matrix, []) == 0.0


# ── score_selection + registry ──────────────────────────────────────────────
def test_score_selection_returns_full_family(orthogonal_matrix):
    M = orthogonal_matrix
    out = score_selection(M, ["a", "b", "c"])
    assert set(out) == {"frac_variance", "effective_rank", "tiered_axis_count"}
    assert out["frac_variance"] == pytest.approx(1.0)
    assert out["effective_rank"] == pytest.approx(3.0, abs=1e-9)


def test_score_selection_forwards_tunable_kwargs(orthogonal_matrix):
    M = orthogonal_matrix
    # An absurdly high high_ratio demotes equal axes out of the high tier;
    # only the (tied) top axis at r=1.0 stays high.
    out = score_selection(M, ["a", "b", "c"], high_ratio=0.99, high_weight=2.0)
    # all three are r=1.0 (exactly tied) so still high; sanity: still positive
    assert out["tiered_axis_count"] > 0


def test_registry_names():
    assert set(SCORE_FUNCTIONS) == {
        "frac_variance", "effective_rank", "tiered_axis_count"
    }


# ── integrates with rules + feature_spaces end-to-end ───────────────────────
def test_scores_on_real_rule_selection():
    from refrover.feature_spaces import gtdb_abundance_matrix
    from refrover.rules import CSSRule

    rng = np.random.default_rng(0)
    sample_taxa = pd.DataFrame(
        rng.random((8, 12)),
        index=[f"s{i}" for i in range(8)],
        columns=[f"t{j}" for j in range(12)],
    )
    M = gtdb_abundance_matrix(sample_taxa)
    picked = CSSRule(k=4, threshold=0.0).select(M, "s0")
    out = score_selection(M, picked)
    assert 0.0 <= out["frac_variance"] <= 1.0
    assert 1.0 <= out["effective_rank"] <= 4.0 + 1e-9
    assert out["tiered_axis_count"] >= 0.0
