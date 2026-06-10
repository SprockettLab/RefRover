"""
Benchmark selector strategies against each other.

Runs each (selector, k) combination, evaluates MAG yield via CheckM2,
and writes a comparison table.
"""

from pathlib import Path
import pandas as pd


def run_benchmark(
    manifest: pd.DataFrame,
    truth_path: Path | str,
    selectors: list[str],
    k_values: list[int],
    outdir: Path | str,
    threads: int = 8,
    checkm2_db: Path | str | None = None,
) -> pd.DataFrame:
    """
    For each (selector, k) combination:
      1. Run RefRoverPipeline with that selector
      2. Run a binner (MetaBAT2) on the coverage output
      3. Evaluate bins with CheckM2

    Returns a summary DataFrame with columns:
      selector, k, n_mags_passing, completeness_mean, contamination_mean,
      cpu_seconds, total_alignment_cpu_seconds
    """
    raise NotImplementedError(
        "Benchmark runner not yet implemented. "
        "Requires CheckM2 installation and a labelled ground-truth community."
    )
