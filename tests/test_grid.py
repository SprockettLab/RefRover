"""Tests for the grid runner (PLAN.md §9 step 5)."""

import numpy as np
import pandas as pd
import pytest

from refrover.grid import (
    FeatureSpaceSpec,
    headline,
    overlap_matrix,
    run_grid,
)
from refrover.rules import RULE_REGISTRY, Rule


def _square_matrix(seed, labels):
    rng = np.random.default_rng(seed)
    n = len(labels)
    M = pd.DataFrame(rng.random((n, n)), index=labels, columns=labels)
    M.index.name = "sample_id"
    M.columns.name = "assembly_id"
    return M


@pytest.fixture
def specs():
    labels = ["s0", "s1", "s2", "s3"]
    return [
        FeatureSpaceSpec("fsA", _square_matrix(1, labels), threshold=0.0),
        FeatureSpaceSpec("fsB", _square_matrix(2, labels), threshold=0.0),
    ]


RULES = ["random", "maxmin", "css", "all_vs_all"]


# ── scores table ────────────────────────────────────────────────────────────
def test_scores_table_shape_and_columns(specs):
    res = run_grid(specs, RULES, k_values=[2, 3])
    # 2 feature spaces × 4 rules × 2 k × 4 samples = 64 rows
    assert len(res.scores) == 64
    assert set(res.scores.columns) == {
        "sample_id", "feature_space", "rule", "k", "n_selected",
        "frac_variance", "effective_rank", "tiered_axis_count",
    }
    assert set(res.scores["feature_space"]) == {"fsA", "fsB"}
    assert set(res.scores["rule"]) == set(RULES)


def test_all_vs_all_is_the_ceiling(specs):
    res = run_grid(specs, RULES, k_values=[3])
    ava = res.scores[res.scores["rule"] == "all_vs_all"]
    assert (ava["n_selected"] == 4).all()                      # everything selected
    assert np.allclose(ava["frac_variance"], 1.0)              # 100% signal line


def test_scores_respect_k_budget(specs):
    res = run_grid(specs, ["css"], k_values=[2])
    css = res.scores[res.scores["rule"] == "css"]
    assert (css["n_selected"] == 2).all()


# ── overlap table ───────────────────────────────────────────────────────────
def test_overlap_pairs_are_unordered_and_complete(specs):
    res = run_grid(specs, RULES, k_values=[2, 3])
    # per (fs, k, sample): C(4,2)=6 pairs; 2 fs × 2 k × 4 samples × 6 = 96
    assert len(res.overlap) == 96
    assert (res.overlap["jaccard"].between(0.0, 1.0)).all()
    # no self-pairs, and rule_a < rule_b (sorted)
    assert (res.overlap["rule_a"] != res.overlap["rule_b"]).all()
    assert (res.overlap["rule_a"] < res.overlap["rule_b"]).all()


def test_overlap_identical_rules_have_jaccard_one():
    # two rules that always return the same set -> jaccard 1.0 everywhere
    labels = ["s0", "s1", "s2", "s3"]
    spec = FeatureSpaceSpec("fs", _square_matrix(3, labels))
    res = run_grid([spec], ["all_vs_all", "css"], k_values=[4])
    # at k=4 (= n), css selects everything too -> identical to all_vs_all
    pair = res.overlap.iloc[0]
    assert pair["jaccard"] == pytest.approx(1.0)


def test_overlap_matrix_pivot_is_symmetric(specs):
    res = run_grid(specs, RULES, k_values=[3])
    mat = overlap_matrix(res.overlap, "fsA", 3)
    assert list(mat.index) == list(mat.columns)
    assert np.allclose(mat.to_numpy(), mat.to_numpy().T)
    assert np.allclose(np.diag(mat.to_numpy()), 1.0)


# ── validity grid: taxonomy_stratified needs clades ─────────────────────────
def test_taxonomy_stratified_skipped_without_clades(specs):
    res = run_grid(specs, ["maxmin", "taxonomy_stratified"], k_values=[2])
    assert "taxonomy_stratified" not in set(res.scores["rule"])
    assert "maxmin" in set(res.scores["rule"])


def test_taxonomy_stratified_runs_with_clades():
    labels = ["s0", "s1", "s2", "s3"]
    clades = pd.Series({"s0": "A", "s1": "A", "s2": "B", "s3": "C"})
    spec = FeatureSpaceSpec("gtdb", _square_matrix(5, labels), clades=clades)
    res = run_grid([spec], ["taxonomy_stratified"], k_values=[3])
    assert "taxonomy_stratified" in set(res.scores["rule"])
    assert len(res.scores) == 4  # one row per sample


# ── optional-dependency rule is skipped grid-wide ───────────────────────────
def test_missing_optional_dependency_skipped(specs, monkeypatch):
    class BoomRule(Rule):
        name = "boom"

        def _rank(self, M, query_id, candidates):
            raise ImportError("pretend the optional package is missing")

    monkeypatch.setitem(RULE_REGISTRY, "boom", BoomRule)
    with pytest.warns(UserWarning, match="unavailable"):
        res = run_grid(specs, ["maxmin", "boom"], k_values=[2])
    assert "boom" not in set(res.scores["rule"])   # skipped everywhere
    assert "maxmin" in set(res.scores["rule"])     # other rules unaffected


# ── headline leaderboard ────────────────────────────────────────────────────
def test_headline_one_row_per_combo_sorted(specs):
    res = run_grid(specs, RULES, k_values=[2, 3])
    h = headline(res.scores)
    # 2 fs × 4 rules × 2 k = 16 combinations
    assert len(h) == 16
    assert (h["tiered_axis_count"].diff().dropna() <= 1e-9).all()  # sorted desc
    assert {"feature_space", "rule", "k", "tiered_axis_count"} <= set(h.columns)


# ── query_ids subsetting ────────────────────────────────────────────────────
def test_query_ids_subsets_samples(specs):
    res = run_grid(specs, ["css"], k_values=[2], query_ids=["s0", "s1"])
    assert set(res.scores["sample_id"]) == {"s0", "s1"}


# ── end-to-end through feature_spaces ───────────────────────────────────────
def test_grid_on_gtdb_bridge():
    from refrover.feature_spaces import gtdb_abundance_matrix

    rng = np.random.default_rng(0)
    sample_taxa = pd.DataFrame(
        rng.random((8, 12)),
        index=[f"s{i}" for i in range(8)],
        columns=[f"t{j}" for j in range(12)],
    )
    M = gtdb_abundance_matrix(sample_taxa)
    spec = FeatureSpaceSpec("gtdb_proxy", M, threshold=0.0)
    res = run_grid([spec], ["random", "maxmin", "css"], k_values=[3, 5])
    assert not res.scores.empty
    assert res.scores["frac_variance"].between(0, 1).all()
