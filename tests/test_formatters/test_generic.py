import pandas as pd
import pytest
from refrover.formatters import format_for_binner


def test_generic_writes_tsv(coverage_df, tmp_path):
    out = format_for_binner(coverage_df, binner="generic", outdir=tmp_path)
    assert out.exists()
    df = pd.read_csv(out, sep="\t", index_col=0)
    assert list(df.columns) == ["length", "s1_depth", "s2_depth", "s3_depth"]
    assert len(df) == 5


def test_generic_skips_existing(coverage_df, tmp_path):
    out1 = format_for_binner(coverage_df, binner="generic", outdir=tmp_path)
    mtime1 = out1.stat().st_mtime
    out2 = format_for_binner(coverage_df, binner="generic", outdir=tmp_path)
    assert out1 == out2
    assert out2.stat().st_mtime == mtime1  # file untouched


def test_generic_force_overwrites(coverage_df, tmp_path):
    import time
    out1 = format_for_binner(coverage_df, binner="generic", outdir=tmp_path)
    mtime1 = out1.stat().st_mtime
    time.sleep(0.01)
    out2 = format_for_binner(coverage_df, binner="generic", outdir=tmp_path, force=True)
    assert out2.stat().st_mtime >= mtime1


def test_unknown_binner_raises(coverage_df, tmp_path):
    with pytest.raises(ValueError, match="Unknown binner"):
        format_for_binner(coverage_df, binner="not_a_binner", outdir=tmp_path)
