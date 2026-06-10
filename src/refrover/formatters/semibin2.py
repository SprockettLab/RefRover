"""SemiBin2 formatter: contig x sample depth matrix, no variance column."""

from pathlib import Path
import pandas as pd


def write_semibin2(coverage_df: pd.DataFrame, outdir: Path, force: bool = False) -> Path:
    out = outdir / "coverage_metabinner.tsv"
    if out.exists() and not force:
        return out
    depth_cols = [c for c in coverage_df.columns if c != "length"]
    coverage_df[depth_cols].to_csv(out, sep="\t")
    return out
