"""
Shared helpers for reading RefRover's normalized coverage schema.

Schema (see refrover.coverage):
  'length'         optional contig length
  '{sample}_depth' mean depth per sample
  '{sample}_var'   depth variance per sample (optional, paired with _depth)
"""

import pandas as pd

DEPTH_SUFFIX = "_depth"
VAR_SUFFIX = "_var"


def mean_columns(df: pd.DataFrame) -> list[str]:
    """Return the per-sample mean-depth columns, in order."""
    return [c for c in df.columns if c.endswith(DEPTH_SUFFIX)]


def sample_of(mean_col: str) -> str:
    """Recover the sample name from a '{sample}_depth' column."""
    return mean_col[: -len(DEPTH_SUFFIX)] if mean_col.endswith(DEPTH_SUFFIX) else mean_col


def var_column(df: pd.DataFrame, mean_col: str) -> str | None:
    """Return the variance column paired with a mean column, or None if absent."""
    candidate = sample_of(mean_col) + VAR_SUFFIX
    return candidate if candidate in df.columns else None
