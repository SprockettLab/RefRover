"""Tests for the unified column-vector selection rules (PLAN.md §5, build step 2)."""

import numpy as np
import pandas as pd
import pytest

from refrover.rules import (
    RULE_REGISTRY,
    AllVsAllRule,
    CSSRule,
    GreedyVarRule,
    KMedoidsRule,
    MaxMinRule,
    RandomRule,
    Rule,
    TaxonomyStratifiedRule,
)


@pytest.fixture
def feature_matrix():
    """
    6 samples × 6 references, uniform `samples × assemblies` shape. Built so the
    reference column vectors fall into two tight groups plus a query column:
        q, a1, a2   ~ "group A" pattern over samples
        b1, b2      ~ "group B" pattern (anti-correlated with A)
        c1          ~ a third independent pattern
    Diagonal-ish self-signal keeps the query's own row above any threshold.
    """
    samples = ["q", "a1", "a2", "b1", "b2", "c1"]
    A = np.array([0.9, 0.8, 0.85, 0.1, 0.15, 0.2])
    B = np.array([0.1, 0.2, 0.15, 0.9, 0.85, 0.2])
    C = np.array([0.2, 0.3, 0.25, 0.3, 0.2, 0.95])
    cols = {
        "q":  A + np.array([0.05, 0, 0, 0, 0, 0]),
        "a1": A * 0.98,
        "a2": A * 1.01,
        "b1": B,
        "b2": B * 0.97,
        "c1": C,
    }
    M = pd.DataFrame(cols, index=samples)
    M.index.name = "sample_id"
    M.columns.name = "assembly_id"
    return M


# ── candidate filter + anchor contract (shared) ─────────────────────────────
def test_candidate_filter_thresholds_on_query_row(feature_matrix):
    rule = MaxMinRule(k=3, threshold=0.5)
    cands = rule._candidates(feature_matrix, "q")
    # query row: q=0.95 high; a1,a2 ~0.8; b1,b2 ~0.1; c1=0.2 -> only A-group clears 0.5
    assert cands[0] == "q"  # anchor first
    assert set(cands) == {"q", "a1", "a2"}


def test_candidate_filter_empty_raises(feature_matrix):
    with pytest.raises(ValueError, match="No candidates"):
        MaxMinRule(k=3, threshold=2.0)._candidates(feature_matrix, "q")


def test_missing_query_raises(feature_matrix):
    with pytest.raises(KeyError):
        MaxMinRule(k=2)._candidates(feature_matrix, "nope")


@pytest.mark.parametrize("name", ["random", "maxmin", "kmedoids", "greedy_var", "css"])
def test_every_rule_anchors_query_and_respects_k(feature_matrix, name):
    rule = RULE_REGISTRY[name](k=3, threshold=0.0)
    result = rule.select(feature_matrix, "q")
    assert result[0] == "q"               # query anchored first
    assert len(result) == 3               # budget respected
    assert len(set(result)) == 3          # no duplicates
    assert set(result) <= set(feature_matrix.columns)


def test_fewer_candidates_than_k_warns_and_returns_all(feature_matrix):
    rule = MaxMinRule(k=5, threshold=0.5)  # only q,a1,a2 eligible
    with pytest.warns(UserWarning, match="Only 3 candidates"):
        result = rule.select(feature_matrix, "q")
    assert set(result) == {"q", "a1", "a2"}


# ── MaxMin spans groups ─────────────────────────────────────────────────────
def test_maxmin_picks_diverse_groups(feature_matrix):
    # k=3 from q-anchor should reach into both the B group and the C pattern,
    # not pile onto the near-duplicate A group.
    result = MaxMinRule(k=3, threshold=0.0).select(feature_matrix, "q")
    assert result[0] == "q"
    assert "a1" not in result and "a2" not in result  # redundant with q, skipped
    assert {"b1", "b2"} & set(result)
    assert "c1" in result


# ── CSS greedily maximises orthogonal-projection variance ───────────────────
def test_css_beats_random_on_variance_explained(feature_matrix):
    from refrover.benchmark import unique_variance_explained

    css = CSSRule(k=3, threshold=0.0).select(feature_matrix, "q")
    rnd = RandomRule(k=3, threshold=0.0, seed=1).select(feature_matrix, "q")
    css_score = unique_variance_explained(feature_matrix, css)
    rnd_score = unique_variance_explained(feature_matrix, rnd)
    assert css_score >= rnd_score
    # CSS reaches each of the 3 independent patterns (A=q, B, C).
    assert "c1" in css and ({"b1", "b2"} & set(css))


def test_css_full_budget_spans_column_space(feature_matrix):
    from refrover.benchmark import unique_variance_explained

    # With k = n, CSS selects a spanning set -> ~all variance explained.
    css = CSSRule(k=6, threshold=0.0).select(feature_matrix, "q")
    assert unique_variance_explained(feature_matrix, css) == pytest.approx(1.0, abs=1e-9)


# ── all_vs_all ceiling ──────────────────────────────────────────────────────
def test_all_vs_all_returns_everything_query_first(feature_matrix):
    result = AllVsAllRule(k=2, threshold=0.9).select(feature_matrix, "q")  # k/thresh ignored
    assert result[0] == "q"
    assert set(result) == set(feature_matrix.columns)


# ── greedy_var / kmedoids smoke ─────────────────────────────────────────────
def test_greedy_var_and_kmedoids_are_deterministic(feature_matrix):
    for rule_cls in (GreedyVarRule, KMedoidsRule):
        r1 = rule_cls(k=3, threshold=0.0).select(feature_matrix, "q")
        r2 = rule_cls(k=3, threshold=0.0).select(feature_matrix, "q")
        assert r1 == r2
        assert r1[0] == "q"


# ── taxonomy_stratified ─────────────────────────────────────────────────────
def test_taxonomy_stratified_one_rep_per_clade(feature_matrix):
    clades = pd.Series(
        {"q": "A", "a1": "A", "a2": "A", "b1": "B", "b2": "B", "c1": "C"}
    )
    rule = TaxonomyStratifiedRule(k=3, threshold=0.0, clades=clades)
    result = rule.select(feature_matrix, "q")
    assert result[0] == "q"
    # one representative per distinct clade among A/B/C -> three distinct clades
    chosen_clades = {clades[r] for r in result}
    assert chosen_clades == {"A", "B", "C"}


def test_taxonomy_stratified_requires_labels():
    with pytest.raises(ValueError, match="requires `clades`"):
        TaxonomyStratifiedRule(k=3)


# ── registry completeness ───────────────────────────────────────────────────
def test_registry_covers_plan_rules():
    expected = {
        "random", "maxmin", "kmedoids", "archetype",
        "greedy_var", "css", "all_vs_all", "taxonomy_stratified",
    }
    assert set(RULE_REGISTRY) == expected
    assert all(issubclass(c, Rule) for c in RULE_REGISTRY.values())


# ── runs on a real feature space (containment-style square matrix) ──────────
def test_rules_run_on_gtdb_bridge_output():
    """End-to-end: a rule consumes a feature_spaces matrix directly."""
    from refrover.feature_spaces import gtdb_abundance_matrix

    rng = np.random.default_rng(0)
    sample_taxa = pd.DataFrame(
        rng.random((5, 8)),
        index=[f"s{i}" for i in range(5)],
        columns=[f"t{j}" for j in range(8)],
    )
    M = gtdb_abundance_matrix(sample_taxa)  # 5×5 samples×assemblies
    result = CSSRule(k=3, threshold=0.0).select(M, "s0")
    assert result[0] == "s0"
    assert len(result) == 3
