import pandas as pd
from refrover.formatters import format_for_binner


def test_metabat2_filename(coverage_df, tmp_path):
    out = format_for_binner(coverage_df, binner="metabat2", outdir=tmp_path)
    assert out.name == "depth.txt"


def test_metabat2_required_columns(coverage_df, tmp_path):
    out = format_for_binner(coverage_df, binner="metabat2", outdir=tmp_path)
    df = pd.read_csv(out, sep="\t")
    assert "contigName" in df.columns
    assert "contigLen" in df.columns
    assert "totalAvgDepth" in df.columns


def test_metabat2_variance_columns(coverage_df, tmp_path):
    out = format_for_binner(coverage_df, binner="metabat2", outdir=tmp_path)
    df = pd.read_csv(out, sep="\t")
    # Each sample has a paired -var column carrying the real variance from the
    # coverage table (not a zero placeholder).
    for sample in ["s1", "s2", "s3"]:
        assert sample in df.columns
        assert f"{sample}-var" in df.columns
        pd.testing.assert_series_equal(
            df[f"{sample}-var"].reset_index(drop=True),
            coverage_df[f"{sample}_var"].reset_index(drop=True),
            check_names=False,
        )


def test_metabat2_variance_falls_back_to_zero_when_absent(coverage_df, tmp_path):
    """A coverage table with no _var columns yields zero variance (back-compat)."""
    means_only = coverage_df[["length", "s1_depth", "s2_depth", "s3_depth"]]
    out = format_for_binner(means_only, binner="metabat2", outdir=tmp_path)
    df = pd.read_csv(out, sep="\t")
    for sample in ["s1", "s2", "s3"]:
        assert (df[f"{sample}-var"] == 0.0).all()


def test_metabat2_total_avg_depth(coverage_df, tmp_path):
    out = format_for_binner(coverage_df, binner="metabat2", outdir=tmp_path)
    df = pd.read_csv(out, sep="\t")
    # totalAvgDepth for first contig: mean of 10.2, 0.0, 7.7 = 5.9666...
    expected = (10.2 + 0.0 + 7.7) / 3
    assert abs(df.loc[0, "totalAvgDepth"] - expected) < 1e-4


def test_metabat2_row_count(coverage_df, tmp_path):
    out = format_for_binner(coverage_df, binner="metabat2", outdir=tmp_path)
    df = pd.read_csv(out, sep="\t")
    assert len(df) == len(coverage_df)


def test_metabat2_nan_depth_filled_with_zero(coverage_df, tmp_path):
    """A NaN depth (no coverage in a co-map sample) must be written as 0.0, not an
    empty field — an empty field crashes MetaBAT2 with `bad_lexical_cast`."""
    import numpy as np
    import pytest

    df_in = coverage_df.copy()
    df_in.loc[df_in.index[0], "s2_depth"] = np.nan
    with pytest.warns(UserWarning, match="filled 1 NaN"):
        out = format_for_binner(df_in, binner="metabat2", outdir=tmp_path)

    # No empty fields anywhere in the written file.
    text = out.read_text()
    assert "\t\t" not in text and not any(
        line.endswith("\t") for line in text.splitlines()
    )
    df = pd.read_csv(out, sep="\t")
    assert df.loc[0, "s2"] == 0.0
    assert df["s2"].notna().all()
