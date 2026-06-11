"""Generic TSV formatter: contig, length, sample1_depth, sample2_depth, ..."""

from pathlib import Path
import pandas as pd

from ._columns import mean_columns


def write_generic(coverage_df: pd.DataFrame, outdir: Path, force: bool = False) -> Path:
    """
    Write a generic coverage TSV: contig index, optional 'length', and one
    mean-depth column per sample. Variance columns are omitted.

    Output: coverage_generic.tsv — tab-separated.
    """
    out = outdir / "coverage_generic.tsv"
    if out.exists() and not force:
        return out
    cols = (["length"] if "length" in coverage_df.columns else []) + mean_columns(coverage_df)
    coverage_df[cols].to_csv(out, sep="\t")
    return out
