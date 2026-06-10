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
    # Each depth column must have a paired -var column set to 0
    depth_cols = ["s1_depth", "s2_depth", "s3_depth"]
    for col in depth_cols:
        assert f"{col}-var" in df.columns
        assert (df[f"{col}-var"] == 0.0).all()


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
