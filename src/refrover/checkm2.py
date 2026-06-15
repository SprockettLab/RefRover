"""
CheckM2 wrapper (Tier-2, PLAN.md §8).

Runs ``checkm2 predict`` on a directory of bin FASTAs and returns the parsed
quality table (one row per bin, with Completeness and Contamination).
`mag_quality.count_mags` turns that into the quality-weighted MAG yield.

CheckM2 is invoked as a subprocess, so it need not share RefRover's Python env —
pass its binary via ``checkm2_path`` (e.g. the binary from a dedicated checkm2
conda env) and its DIAMOND database via ``db_path``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pandas as pd

QUALITY_REPORT = "quality_report.tsv"


def run_checkm2(
    bins_dir: Path | str,
    outdir: Path | str,
    *,
    db_path: Path | str | None = None,
    extension: str = "fa",
    threads: int = 8,
    force: bool = False,
    checkm2_path: str = "checkm2",
) -> pd.DataFrame:
    """
    Assess bin quality with CheckM2.

    Parameters
    ----------
    bins_dir : path
        Directory of bin FASTAs (MetaBAT2 output).
    outdir : path
        CheckM2 output directory; the report lands at ``outdir/quality_report.tsv``.
    db_path : path, optional
        CheckM2 DIAMOND database (``--database_path``). If None, CheckM2 uses its
        configured default.
    extension : str
        Bin file extension (``--extension``).

    Returns
    -------
    pd.DataFrame
        The parsed ``quality_report.tsv`` (columns include Name, Completeness,
        Contamination). Idempotent: an existing report is loaded rather than
        recomputed unless ``force``. Raises RuntimeError on a non-zero exit.
    """
    outdir = Path(outdir)
    report = outdir / QUALITY_REPORT
    if report.exists() and not force:
        return load_quality_report(report)

    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        checkm2_path, "predict",
        "--input", str(bins_dir),
        "--output-directory", str(outdir),
        "--extension", extension,
        "--threads", str(threads),
    ]
    if db_path is not None:
        cmd += ["--database_path", str(db_path)]
    if force:
        cmd.append("--force")

    # When checkm2_path is absolute (from a separate conda env), its sibling
    # tools (prodigal, diamond) live in the same bin/ directory but aren't on
    # the caller's PATH. Prepend that directory so checkm2 can find them.
    env = os.environ.copy()
    checkm2_bin = Path(checkm2_path)
    if checkm2_bin.is_absolute():
        env["PATH"] = str(checkm2_bin.parent) + os.pathsep + env.get("PATH", "")

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"CheckM2 failed:\n{result.stderr}")
    if not report.exists():
        raise RuntimeError(
            f"CheckM2 finished but no {QUALITY_REPORT} at {report}"
        )
    return load_quality_report(report)


def load_quality_report(path: Path | str) -> pd.DataFrame:
    """Load a CheckM2 ``quality_report.tsv`` into a DataFrame."""
    return pd.read_csv(path, sep="\t")


__all__ = ["run_checkm2", "load_quality_report", "QUALITY_REPORT"]
