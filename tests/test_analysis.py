"""Tests for the grid analysis (PLAN.md §7, build step 6)."""

import numpy as np
import pandas as pd
import pytest

from refrover.analysis import (
    divergence,
    method_advantage,
    win_rates,
    winner_feature_contrast,
    winning_method,
)


@pytest.fixture
def scores():
    """
    Small hand-built score table: one feature space, k=5, three samples, four
    rules. Designed so css wins s0 & s1, maxmin wins s2; random is always worst;
    all_vs_all is the (excluded) ceiling.
    """
    rows = [
        # sample, rule, tiered
        ("s0", "random", 1.0), ("s0", "maxmin", 2.0), ("s0", "css", 3.0), ("s0", "all_vs_all", 9.0),
        ("s1", "random", 1.0), ("s1", "maxmin", 1.5), ("s1", "css", 2.5), ("s1", "all_vs_all", 9.0),
        ("s2", "random", 1.0), ("s2", "maxmin", 4.0), ("s2", "css", 2.0), ("s2", "all_vs_all", 9.0),
    ]
    return pd.DataFrame(
        [
            {"feature_space": "fs", "k": 5, "sample_id": s, "rule": r,
             "tiered_axis_count": t, "frac_variance": 0.5, "effective_rank": 1.0,
             "n_selected": 5}
            for s, r, t in rows
        ]
    )


@pytest.fixture
def features():
    f = pd.DataFrame(
        {"species_distinctness": [0.9, 0.8, 0.2],
         "pielou_evenness": [0.2, 0.3, 0.9]},
        index=["s0", "s1", "s2"],
    )
    f.index.name = "sample_id"
    return f


# ── winning_method ──────────────────────────────────────────────────────────
def test_winning_method_excludes_ceiling(scores):
    w = winning_method(scores)
    assert set(w["winner"]) == {"css", "maxmin"}     # all_vs_all excluded
    got = dict(zip(w["sample_id"], w["winner"]))
    assert got == {"s0": "css", "s1": "css", "s2": "maxmin"}


def test_winning_method_one_row_per_sample(scores):
    w = winning_method(scores)
    assert len(w) == 3
    assert set(w.columns) == {"feature_space", "k", "sample_id", "winner", "tiered_axis_count"}


# ── win_rates ───────────────────────────────────────────────────────────────
def test_win_rates_sum_to_one(scores):
    wr = win_rates(winning_method(scores))
    assert wr["win_rate"].sum() == pytest.approx(1.0)
    css = wr[wr["winner"] == "css"].iloc[0]
    assert css["n_wins"] == 2 and css["win_rate"] == pytest.approx(2 / 3)


# ── method_advantage ────────────────────────────────────────────────────────
def test_method_advantage_vs_random(scores):
    adv = method_advantage(scores)
    assert "random" not in set(adv["rule"])      # baseline dropped
    assert "all_vs_all" not in set(adv["rule"])  # ceiling dropped
    # css on s0: 3.0 − random 1.0 = 2.0
    css_s0 = adv[(adv["rule"] == "css") & (adv["sample_id"] == "s0")]["advantage"].iloc[0]
    assert css_s0 == pytest.approx(2.0)


# ── winner_feature_contrast ─────────────────────────────────────────────────
def test_winner_feature_contrast_separates_winners(scores, features):
    c = winner_feature_contrast(winning_method(scores), features)
    css_row = c[c["winner"] == "css"].iloc[0]
    maxmin_row = c[c["winner"] == "maxmin"].iloc[0]
    # css wins the distinct, uneven samples; maxmin wins the diverse, even one
    assert css_row["species_distinctness"] > maxmin_row["species_distinctness"]
    assert css_row["pielou_evenness"] < maxmin_row["pielou_evenness"]


# ── divergence ──────────────────────────────────────────────────────────────
def test_divergence_orders_most_divergent_first():
    overlap = pd.DataFrame(
        [
            # s0: rules agree a lot; s1: rules diverge
            {"feature_space": "fs", "k": 5, "sample_id": "s0", "rule_a": "a", "rule_b": "b", "jaccard": 0.9},
            {"feature_space": "fs", "k": 5, "sample_id": "s0", "rule_a": "a", "rule_b": "c", "jaccard": 0.8},
            {"feature_space": "fs", "k": 5, "sample_id": "s1", "rule_a": "a", "rule_b": "b", "jaccard": 0.1},
            {"feature_space": "fs", "k": 5, "sample_id": "s1", "rule_a": "a", "rule_b": "c", "jaccard": 0.2},
        ]
    )
    d = divergence(overlap)
    assert list(d["sample_id"]) == ["s1", "s0"]                 # most divergent first
    assert d.iloc[0]["mean_agreement"] == pytest.approx(0.15)


# ── fit_winner_rule (optional sklearn) ──────────────────────────────────────
def test_fit_winner_rule_produces_text_and_cv():
    sklearn = pytest.importorskip("sklearn")  # noqa: F841
    from refrover.analysis import fit_winner_rule

    # 40 samples, cleanly separable: css when distinctness high, maxmin when low.
    rng = np.random.default_rng(0)
    dist = rng.random(40)
    winners = pd.DataFrame(
        {
            "feature_space": "fs", "k": 5,
            "sample_id": [f"s{i}" for i in range(40)],
            "winner": ["css" if d > 0.5 else "maxmin" for d in dist],
        }
    )
    features = pd.DataFrame(
        {"species_distinctness": dist, "noise": rng.random(40)},
        index=[f"s{i}" for i in range(40)],
    )
    out = fit_winner_rule(winners, features, feature_space="fs", k=5, max_depth=2)
    assert "species_distinctness" in out["rules"]
    assert out["train_accuracy"] == pytest.approx(1.0)
    # cleanly separable -> CV accuracy is high and clears the majority baseline
    assert out["cv_accuracy"] is not None and out["cv_accuracy"] > 0.9
    assert out["lift"] > 0.0
    assert out["cv_folds"] == 5
    assert set(out["classes"]) == {"css", "maxmin"}


def test_fit_winner_rule_cv_beats_no_baseline_for_noise():
    """When the winner is unrelated to features, CV accuracy must NOT beat baseline."""
    sklearn = pytest.importorskip("sklearn")  # noqa: F841
    from refrover.analysis import fit_winner_rule

    rng = np.random.default_rng(1)
    # winner is random coin flips; features are pure noise -> no real signal
    y = rng.integers(0, 2, 40)
    winners = pd.DataFrame(
        {
            "feature_space": "fs", "k": 5,
            "sample_id": [f"s{i}" for i in range(40)],
            "winner": ["css" if v else "maxmin" for v in y],
        }
    )
    features = pd.DataFrame(
        {"f1": rng.random(40), "f2": rng.random(40)},
        index=[f"s{i}" for i in range(40)],
    )
    out = fit_winner_rule(winners, features, feature_space="fs", k=5, max_depth=3)
    # train overfits (gap), but honest CV lift should be ~0 or negative
    assert out["train_accuracy"] > out["cv_accuracy"]
    assert out["lift"] <= 0.1


def test_fit_winner_rule_cv_skipped_for_tiny_class():
    sklearn = pytest.importorskip("sklearn")  # noqa: F841
    from refrover.analysis import fit_winner_rule

    # 'css' appears once -> can't stratify into >=2 folds
    winners = pd.DataFrame(
        {
            "feature_space": "fs", "k": 5,
            "sample_id": [f"s{i}" for i in range(6)],
            "winner": ["maxmin"] * 5 + ["css"],
        }
    )
    features = pd.DataFrame(
        {"x": np.arange(6, dtype=float)}, index=[f"s{i}" for i in range(6)]
    )
    out = fit_winner_rule(winners, features, feature_space="fs", k=5)
    assert out["cv_accuracy"] is None and out["cv_folds"] == 0
    assert out["baseline_accuracy"] == pytest.approx(5 / 6)


def test_fit_winner_rule_single_winner():
    sklearn = pytest.importorskip("sklearn")  # noqa: F841
    from refrover.analysis import fit_winner_rule

    winners = pd.DataFrame(
        {"feature_space": "fs", "k": 5, "sample_id": ["s0", "s1"], "winner": ["css", "css"]}
    )
    features = pd.DataFrame({"x": [0.1, 0.2]}, index=["s0", "s1"])
    out = fit_winner_rule(winners, features, feature_space="fs", k=5)
    assert "only one winner" in out["rules"]
    assert out["classes"] == ["css"]
    assert out["cv_accuracy"] is None
