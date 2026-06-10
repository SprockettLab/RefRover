from .registry import BINNER_REGISTRY
from pathlib import Path
import pandas as pd


def format_for_binner(
    coverage_df: pd.DataFrame,
    binner: str,
    outdir: Path | str,
    force: bool = False,
):
    """Dispatch coverage_df to the formatter for the requested binner."""
    outdir = Path(outdir)
    if binner not in BINNER_REGISTRY:
        raise ValueError(
            f"Unknown binner '{binner}'. Available: {sorted(BINNER_REGISTRY)}"
        )
    return BINNER_REGISTRY[binner](coverage_df, outdir, force=force)
