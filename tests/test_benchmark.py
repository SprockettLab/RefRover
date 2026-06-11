"""Tests for the Tier-1 variance-explained selector ranking."""

import numpy as np
import pandas as pd
import pytest

from refrover.benchmark import unique_variance_explained, rank_selectors


@pytest.fixture
def feature_matrix():
    """6 samples × 4 feature columns with structure (not full rank in practice)."""
    rng = np.random.default_rng(0)
    cols = ["a", "b", "c", "d"]
    data = rng.normal(size=(6, 4))
    # make 'd' a near-duplicate of 'a' (correlated/redundant)
    data[:, 3] = data[:, 0] + 1e-3 * rng.normal(size=6)
    return pd.DataFrame(data, index=[f"s{i}" for i in range(6)], columns=cols)


def test_all_columns_explain_everything(feature_matrix):
    ve = unique_variance_explained(feature_matrix, list(feature_matrix.columns))
    assert ve == pytest.approx(1.0, abs=1e-9)


def test_empty_selection_explains_nothing(feature_matrix):
    assert unique_variance_explained(feature_matrix, []) == 0.0


def test_value_in_unit_interval(feature_matrix):
    ve = unique_variance_explained(feature_matrix, ["a", "b"])
    assert 0.0 <= ve <= 1.0


def test_redundant_column_adds_little(feature_matrix):
    """'d' duplicates 'a', so adding it explains almost no new variance."""
    base = unique_variance_explained(feature_matrix, ["a"])
    plus_dupe = unique_variance_explained(feature_matrix, ["a", "d"])
    plus_new = unique_variance_explained(feature_matrix, ["a", "b"])
    assert plus_dupe - base < plus_new - base


def test_more_columns_never_explain_less(feature_matrix):
    """Orthogonal projection: a superset never explains less (nested sets)."""
    small = unique_variance_explained(feature_matrix, ["a", "b"])
    big = unique_variance_explained(feature_matrix, ["a", "b", "c"])
    assert big >= small - 1e-9


# ── rank_selectors ─────────────────────────────────────────────────────────────

def test_rank_returns_expected_columns(clustered_sim):
    ranking = rank_selectors(
        clustered_sim, selectors=["random", "maxmin"], k_values=[3, 4],
        min_similarity=0.1,
    )
    assert list(ranking.columns) == [
        "selector", "k", "mean_var_explained", "median_var_explained", "n_samples",
    ]
    assert set(ranking["selector"]) == {"random", "maxmin"}
    assert set(ranking["k"]) == {3, 4}


def test_rank_maxmin_monotone_in_k(clustered_sim):
    """MaxMin is greedily nested, so more prototypes explain at least as much."""
    ranking = rank_selectors(
        clustered_sim, selectors=["maxmin"], k_values=[3, 4], min_similarity=0.1,
    )
    by_k = ranking.set_index("k")["mean_var_explained"]
    assert by_k[4] >= by_k[3] - 1e-9


def test_rank_skips_feedback_with_warning(clustered_sim):
    with pytest.warns(UserWarning):
        ranking = rank_selectors(
            clustered_sim, selectors=["maxmin", "feedback"], k_values=[3],
            min_similarity=0.1,
        )
    assert set(ranking["selector"]) == {"maxmin"}


def test_rank_is_reproducible(clustered_sim):
    """The random baseline is seeded, so the ranking is deterministic."""
    a = rank_selectors(clustered_sim, selectors=["random"], k_values=[3], min_similarity=0.1)
    b = rank_selectors(clustered_sim, selectors=["random"], k_values=[3], min_similarity=0.1)
    pd.testing.assert_frame_equal(a, b)


def test_rank_on_containment_matrix(clustered_sim):
    """All selectors, including containment, run on a single shared matrix."""
    ranking = rank_selectors(
        clustered_sim, selectors=["maxmin", "containment"], k_values=[3],
        min_similarity=0.1,
    )
    assert set(ranking["selector"]) == {"maxmin", "containment"}
    assert (ranking["mean_var_explained"] >= 0).all()
