"""CONCOCT formatter: contig x sample matrix, tab-separated, integer depths."""

from pathlib import Path
import pandas as pd

from ._columns import mean_columns


def write_concoct(coverage_df: pd.DataFrame, outdir: Path, force: bool = False) -> Path:
    out = outdir / "coverage_table.tsv"
    if out.exists() and not force:
        return out
    df_out = coverage_df[mean_columns(coverage_df)].round(0).astype(int)
    df_out.to_csv(out, sep="\t")
    return out
