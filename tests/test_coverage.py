"""Tests for CoverM output normalization into RefRover's internal schema."""

import pandas as pd

from refrover.coverage import normalize_coverage, load_coverage


def _raw_coverm_frame():
    """Mimic a raw `coverm contig --methods length mean variance` table.

    CoverM names columns `{stoit} {Method}` and repeats Length per BAM.
    """
    return pd.DataFrame(
        {
            "S1.sorted Length": [1000, 2000, 500],
            "S1.sorted Mean": [10.2, 0.0, 5.5],
            "S1.sorted Variance": [2.1, 0.0, 1.2],
            "S2.sorted Length": [1000, 2000, 500],
            "S2.sorted Mean": [0.0, 15.3, 6.1],
            "S2.sorted Variance": [0.0, 3.3, 1.5],
        },
        index=pd.Index(["contig_0", "contig_1", "contig_2"], name="Contig"),
    )


def test_normalize_maps_methods_to_schema():
    out = normalize_coverage(_raw_coverm_frame())
    assert list(out.columns) == ["length", "S1_depth", "S2_depth", "S1_var", "S2_var"]
    assert out.loc["contig_0", "S1_depth"] == 10.2
    assert out.loc["contig_1", "S2_var"] == 3.3


def test_normalize_collapses_repeated_length():
    out = normalize_coverage(_raw_coverm_frame())
    assert (out["length"] == [1000, 2000, 500]).all()
    assert out["length"].dtype.kind == "i"  # integer length


def test_normalize_handles_missing_length():
    raw = _raw_coverm_frame().drop(columns=["S1.sorted Length", "S2.sorted Length"])
    out = normalize_coverage(raw)
    assert "length" not in out.columns
    assert list(out.columns) == ["S1_depth", "S2_depth", "S1_var", "S2_var"]


def test_normalize_passthrough_when_already_normalized(coverage_df):
    """An already-normalized frame (has _depth columns) is returned unchanged."""
    out = normalize_coverage(coverage_df)
    assert out is coverage_df


def test_load_coverage_normalizes_from_disk(tmp_path):
    raw = _raw_coverm_frame()
    p = tmp_path / "coverage.tsv"
    raw.to_csv(p, sep="\t")
    out = load_coverage(p)
    assert "S1_depth" in out.columns
    assert "S1_var" in out.columns
    assert out.loc["contig_2", "S1_depth"] == 5.5


def test_normalize_mean_without_variance():
    """Mean-only CoverM output still normalizes (no _var columns produced)."""
    raw = pd.DataFrame(
        {"S1.sorted Mean": [3.0, 4.0], "S2.sorted Mean": [1.0, 2.0]},
        index=pd.Index(["c0", "c1"], name="Contig"),
    )
    out = normalize_coverage(raw)
    assert list(out.columns) == ["S1_depth", "S2_depth"]
