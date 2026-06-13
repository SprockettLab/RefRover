"""
MetaBAT2 binning wrapper (Tier-2, PLAN.md §8).

Thin subprocess wrapper following the `coverage.run_coverm` pattern: takes a
focal assembly FASTA and a MetaBAT2-format depth file (written by
`formatters.metabat2.write_metabat2`) and produces bin FASTAs. The Tier-2 harness
then hands the bins to CheckM2.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def run_metabat2(
    contigs_fasta: Path | str,
    depth_txt: Path | str,
    outdir: Path | str,
    *,
    min_contig: int = 1500,
    threads: int = 8,
    force: bool = False,
    metabat2_path: str = "metabat2",
) -> Path:
    """
    Bin a focal assembly's contigs with MetaBAT2.

    Parameters
    ----------
    contigs_fasta : path
        The focal assembly (the contig set being binned).
    depth_txt : path
        jgi_summarize_bam_contig_depths-format depth (from `write_metabat2`).
    outdir : path
        Directory to write bins into; bins are ``outdir/bin.<n>.fa``.
    min_contig : int
        MetaBAT2 ``-m`` (must be ≥ 1500).
    threads : int
        MetaBAT2 ``-t``.

    Returns
    -------
    Path
        ``outdir`` (the bins directory). Idempotent: if it already holds ``*.fa``
        and ``force`` is False, MetaBAT2 is not re-run. Raises RuntimeError on a
        non-zero exit.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not force and any(outdir.glob("*.fa")):
        return outdir

    cmd = [
        metabat2_path,
        "-i", str(contigs_fasta),
        "-a", str(depth_txt),
        "-o", str(outdir / "bin"),
        "-m", str(min_contig),
        "-t", str(threads),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"MetaBAT2 failed:\n{result.stderr}")
    return outdir


def list_bins(bins_dir: Path | str, *, extension: str = "fa") -> list[Path]:
    """Sorted list of bin FASTAs in a MetaBAT2 output directory."""
    return sorted(Path(bins_dir).glob(f"*.{extension}"))


__all__ = ["run_metabat2", "list_bins"]
