"""
CoverM wrapper: compute per-contig mean depth from sorted BAMs.
"""

import subprocess
from pathlib import Path
import pandas as pd


def run_coverm(
    bams_dir: Path | str,
    outdir: Path | str,
    *,
    threads: int = 8,
    force: bool = False,
    coverm_path: str = "coverm",
) -> Path:
    """
    Run CoverM on all BAMs in bams_dir.

    Writes coverage.tsv to outdir with columns:
      Contig  Length  sample1_depth  sample2_depth ...

    Returns path to coverage.tsv.
    """
    bams_dir = Path(bams_dir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_tsv = outdir / "coverage.tsv"
    if out_tsv.exists() and not force:
        return out_tsv

    bam_files = sorted(bams_dir.glob("*.sorted.bam"))
    if not bam_files:
        raise FileNotFoundError(f"No *.sorted.bam files found in {bams_dir}")

    cmd = [
        coverm_path, "contig",
        "--bam-files", *[str(b) for b in bam_files],
        "--methods", "mean",
        "--threads", str(threads),
        "--output-file", str(out_tsv),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"CoverM failed:\n{result.stderr}")

    return out_tsv


def load_coverage(tsv_path: Path | str) -> pd.DataFrame:
    """Load a CoverM coverage.tsv into a DataFrame indexed by contig name."""
    df = pd.read_csv(tsv_path, sep="\t", index_col=0)
    return df
