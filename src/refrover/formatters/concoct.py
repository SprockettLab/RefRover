"""CONCOCT formatter: contig x sample matrix, tab-separated, integer depths."""

from pathlib import Path
import pandas as pd


def write_concoct(coverage_df: pd.DataFrame, outdir: Path, force: bool = False) -> Path:
    out = outdir / "coverage_table.tsv"
    if out.exists() and not force:
        return out
    depth_cols = [c for c in coverage_df.columns if c != "length"]
    df_out = coverage_df[depth_cols].round(0).astype(int)
    df_out.to_csv(out, sep="\t")
    return out
