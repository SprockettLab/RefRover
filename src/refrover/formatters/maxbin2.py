"""MaxBin2 formatter: one .abund file per sample, single depth column."""

from pathlib import Path
import pandas as pd

from ._columns import mean_columns, sample_of


def write_maxbin2(coverage_df: pd.DataFrame, outdir: Path, force: bool = False) -> list[Path]:
    out_paths = []
    for col in mean_columns(coverage_df):
        sample = sample_of(col)
        out = outdir / f"{sample}.abund"
        out_paths.append(out)
        if out.exists() and not force:
            continue
        coverage_df[[col]].to_csv(out, sep="\t", header=False)
    return out_paths
