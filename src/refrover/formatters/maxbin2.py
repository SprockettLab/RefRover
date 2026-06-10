"""MaxBin2 formatter: one .abund file per sample, single depth column."""

from pathlib import Path
import pandas as pd


def write_maxbin2(coverage_df: pd.DataFrame, outdir: Path, force: bool = False) -> list[Path]:
    depth_cols = [c for c in coverage_df.columns if c != "length"]
    out_paths = []
    for col in depth_cols:
        out = outdir / f"{col}.abund"
        out_paths.append(out)
        if out.exists() and not force:
            continue
        coverage_df[[col]].to_csv(out, sep="\t", header=False)
    return out_paths
