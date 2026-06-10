"""Generic TSV formatter: contig, length, sample1_depth, sample2_depth, ..."""

from pathlib import Path
import pandas as pd


def write_generic(coverage_df: pd.DataFrame, outdir: Path, force: bool = False) -> Path:
    """
    Write a generic coverage TSV.

    Input: DataFrame with index=contig, columns including 'length' and one depth
    column per sample (e.g. 's001_depth').

    Output: coverage_generic.tsv  — same layout, tab-separated.
    """
    out = outdir / "coverage_generic.tsv"
    if out.exists() and not force:
        return out
    coverage_df.to_csv(out, sep="\t")
    return out
