"""Tests for per-sample feature computation (PLAN.md §7.1, build step 4)."""

import numpy as np
import pandas as pd
import pytest

from refrover.features import (
    candidate_features,
    diversity_features,
    feature_table,
    per_sample_features,
    saturation_curve,
    saturation_features,
    species_distinctness,
)


# ── diversity_features ──────────────────────────────────────────────────────
def test_diversity_even_community():
    # 4 taxa, perfectly even -> max Shannon, evenness 1, low dominance
    a = pd.Series([0.25, 0.25, 0.25, 0.25], index=list("abcd"))
    d = diversity_features(a)
    assert d["richness"] == 4
    assert d["shannon"] == pytest.approx(np.log(4))
    assert d["pielou_evenness"] == pytest.approx(1.0)
    assert d["simpson_dominance"] == pytest.approx(0.25)
    assert d["berger_parker"] == pytest.approx(0.25)


def test_diversity_dominated_community():
    # one dominant taxon -> low evenness, high dominance
    a = pd.Series([0.97, 0.01, 0.01, 0.01], index=list("abcd"))
    d = diversity_features(a)
    assert d["richness"] == 4
    assert d["pielou_evenness"] < 0.3
    assert d["berger_parker"] == pytest.approx(0.97)


def test_diversity_ignores_absent_taxa():
    a = pd.Series([0.5, 0.5, 0.0, 0.0], index=list("abcd"))
    assert diversity_features(a)["richness"] == 2


def test_diversity_empty_sample():
    a = pd.Series([0.0, 0.0], index=list("ab"))
    d = diversity_features(a)
    assert d["richness"] == 0 and d["shannon"] == 0.0


# ── species_distinctness ────────────────────────────────────────────────────
def test_species_distinctness_independent_vs_redundant():
    # taxa t1,t2 co-vary perfectly (redundant); t3 is independent.
    samples = [f"s{i}" for i in range(6)]
    base = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    A = pd.DataFrame(
        {
            "t1": base,
            "t2": base * 2,          # perfectly correlated with t1
            "t3": base[::-1],        # anti-correlated -> distinct
        },
        index=samples,
    )
    # query present-taxa = all three (positive everywhere)
    qrow = pd.Series([0.3, 0.3, 0.4], index=["t1", "t2", "t3"], name="q")
    A2 = pd.concat([A, pd.DataFrame([qrow.values], index=["q"], columns=A.columns)])
    d_all = species_distinctness(A2, "q")
    # with an anti-correlated taxon present, mean distance is clearly > 0
    assert d_all > 0.5


def test_species_distinctness_single_taxon_is_zero():
    A = pd.DataFrame({"t1": [1.0, 2.0], "t2": [0.0, 0.0]}, index=["q", "s1"])
    assert species_distinctness(A, "q") == 0.0


# ── candidate_features ──────────────────────────────────────────────────────
@pytest.fixture
def feature_matrix():
    samples = ["q", "a1", "a2", "b1", "c1"]
    A = np.array([0.9, 0.8, 0.85, 0.1, 0.2])
    B = np.array([0.1, 0.2, 0.15, 0.9, 0.2])
    C = np.array([0.2, 0.3, 0.25, 0.3, 0.95])
    M = pd.DataFrame(
        {"q": A, "a1": A * 0.98, "a2": A * 1.01, "b1": B, "c1": C}, index=samples
    )
    M.index.name = "sample_id"
    M.columns.name = "assembly_id"
    return M


def test_candidate_features_threshold_counts(feature_matrix):
    # query row: q=0.9, a1≈0.88, a2≈0.91, b1=0.1, c1=0.2 -> threshold 0.5 keeps q,a1,a2
    cf = candidate_features(feature_matrix, "q", threshold=0.5)
    assert cf["n_candidates"] == 3
    assert cf["mean_candidate_signal"] > 0.8
    assert 0.0 <= cf["candidate_separability"] <= 2.0


def test_candidate_features_no_threshold_uses_all(feature_matrix):
    cf = candidate_features(feature_matrix, "q", threshold=0.0)
    assert cf["n_candidates"] == 5


# ── saturation curve + shape ────────────────────────────────────────────────
def test_saturation_curve_monotone_cumulative(feature_matrix):
    curve = saturation_curve(feature_matrix, "q", threshold=0.0, k_max=5)
    cum = curve["cumulative_var"].to_numpy()
    assert (np.diff(cum) >= -1e-9).all()         # cumulative non-decreasing
    assert curve["marginal_var"].iloc[1] <= curve["marginal_var"].iloc[0] + 1e-9
    assert cum[-1] == pytest.approx(1.0, abs=1e-9)  # full budget spans space


def test_saturation_features_shapes(feature_matrix):
    feats = saturation_features(saturation_curve(feature_matrix, "q", k_max=5))
    assert 0.0 <= feats["sat_auc"] <= 1.0
    assert feats["sat_k_to_90"] >= 1
    assert feats["sat_early_decay"] >= 0.0


def test_saturation_fast_vs_slow_auc():
    # Uniform matrices are square & same-labelled (sample ↔ assembly 1:1).
    labels = ["q", "r1", "r2", "r3", "r4"]
    # Fast-saturating: one dominant axis + near-duplicates -> high AUC, low k_to_90.
    dom = np.array([0.9, 0.1, 0.85, 0.15, 0.5])
    fast = pd.DataFrame(
        {"q": dom, "r1": dom * 0.99, "r2": dom * 1.01, "r3": dom * 0.98, "r4": dom * 1.02},
        index=labels,
    )
    # Slow-saturating: several independent axes -> lower AUC, higher k_to_90.
    rng = np.random.default_rng(0)
    slow = pd.DataFrame(
        {c: rng.random(5) for c in labels}, index=labels
    )
    f_fast = saturation_features(saturation_curve(fast, "q", k_max=5))
    f_slow = saturation_features(saturation_curve(slow, "q", k_max=5))
    assert f_fast["sat_auc"] >= f_slow["sat_auc"]
    assert f_fast["sat_k_to_90"] <= f_slow["sat_k_to_90"]


# ── assembled table ─────────────────────────────────────────────────────────
def test_per_sample_features_with_and_without_abundance(feature_matrix):
    without = per_sample_features(feature_matrix, "q", threshold=0.0)
    assert "n_candidates" in without and "sat_auc" in without
    assert "richness" not in without  # no abundance -> no diversity features

    abundance = pd.DataFrame(
        np.abs(np.random.default_rng(1).random((5, 6))),
        index=feature_matrix.index,
        columns=[f"t{j}" for j in range(6)],
    )
    with_ab = per_sample_features(feature_matrix, "q", abundance=abundance, threshold=0.0)
    assert {"richness", "shannon", "species_distinctness"} <= set(with_ab)


def test_feature_table_one_row_per_sample(feature_matrix):
    abundance = pd.DataFrame(
        np.abs(np.random.default_rng(2).random((5, 6))),
        index=feature_matrix.index,
        columns=[f"t{j}" for j in range(6)],
    )
    table = feature_table(feature_matrix, abundance=abundance, threshold=0.0, k_max=5)
    assert list(table.index) == list(feature_matrix.index)
    assert table.index.name == "sample_id"
    for col in ["n_candidates", "candidate_separability", "sat_auc",
                "richness", "shannon", "species_distinctness"]:
        assert col in table.columns
    assert not table[["n_candidates", "sat_auc"]].isna().any().any()


# ── integrates with a real feature space ────────────────────────────────────
def test_feature_table_on_gtdb_bridge():
    from refrover.feature_spaces import gtdb_abundance_matrix

    rng = np.random.default_rng(0)
    sample_taxa = pd.DataFrame(
        rng.random((8, 10)),
        index=[f"s{i}" for i in range(8)],
        columns=[f"t{j}" for j in range(10)],
    )
    M = gtdb_abundance_matrix(sample_taxa)  # 8×8
    table = feature_table(M, abundance=sample_taxa, threshold=0.0, k_max=5)
    assert table.shape[0] == 8
    assert (table["n_candidates"] == 8).all()
