"""SemiBin2 formatter: contig x sample depth matrix, no variance column."""

from pathlib import Path
import pandas as pd

from ._columns import mean_columns


def write_semibin2(coverage_df: pd.DataFrame, outdir: Path, force: bool = False) -> Path:
    out = outdir / "coverage_metabinner.tsv"
    if out.exists() and not force:
        return out
    coverage_df[mean_columns(coverage_df)].to_csv(out, sep="\t")
    return out
