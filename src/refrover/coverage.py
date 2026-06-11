"""
CoverM wrapper: compute per-contig depth (mean + variance) from sorted BAMs.

Internal coverage schema
------------------------
Everything downstream of `load_coverage` uses one normalized DataFrame:

  index            contig name
  column 'length'  contig length (int), when available
  '{sample}_depth' mean depth for that sample
  '{sample}_var'   depth variance for that sample (paired with _depth)

`load_coverage` parses CoverM's raw `{sample} {Method}` columns into this schema.
The MetaBAT2 formatter consumes the variance columns; the others use only the
means. Keeping mean and variance together is what lets MetaBAT2 emit a real
per-sample variance instead of a placeholder.
"""

import subprocess
from pathlib import Path
import pandas as pd

DEPTH_SUFFIX = "_depth"
VAR_SUFFIX = "_var"
DEFAULT_METHODS = ("length", "mean", "variance")


def run_coverm(
    bams_dir: Path | str,
    outdir: Path | str,
    *,
    threads: int = 8,
    force: bool = False,
    methods: tuple[str, ...] = DEFAULT_METHODS,
    coverm_path: str = "coverm",
) -> Path:
    """
    Run CoverM on all sorted BAMs in bams_dir.

    Writes the raw CoverM table to outdir/coverage.tsv. By default requests
    `length mean variance` so the MetaBAT2 formatter has a real variance column.
    Use `load_coverage` to read the result into the normalized schema.

    Returns the path to coverage.tsv.
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
        "--methods", *methods,
        "--threads", str(threads),
        "--output-file", str(out_tsv),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"CoverM failed:\n{result.stderr}")

    return out_tsv


def _clean_sample(stoit: str) -> str:
    """Strip CoverM's BAM-derived decorations to recover the sample name."""
    name = stoit.strip()
    for suffix in (".bam", ".sorted", ".sort"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name.strip()


def normalize_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a raw CoverM table into RefRover's internal coverage schema.

    CoverM names columns `{sample} {Method}` (e.g. `S1.sorted Mean`,
    `S1.sorted Variance`, `S1.sorted Length`). This maps:
      ` Mean`     -> `{sample}_depth`
      ` Variance` -> `{sample}_var`
      ` Length`   -> a single `length` column (CoverM repeats it per BAM)

    If the frame already looks normalized (has any `_depth` column), it is
    returned unchanged — so re-reading an already-normalized table is a no-op.
    """
    if any(c.endswith(DEPTH_SUFFIX) for c in df.columns):
        return df

    means: dict[str, pd.Series] = {}
    variances: dict[str, pd.Series] = {}
    length: pd.Series | None = None

    for col in df.columns:
        low = col.lower()
        if low.endswith(" mean"):
            sample = _clean_sample(col[: -len(" Mean")])
            means[f"{sample}{DEPTH_SUFFIX}"] = df[col]
        elif low.endswith(" variance"):
            sample = _clean_sample(col[: -len(" Variance")])
            variances[f"{sample}{VAR_SUFFIX}"] = df[col]
        elif low.endswith(" length") or low == "length":
            if length is None:
                length = df[col]

    out = pd.DataFrame(index=df.index)
    if length is not None:
        out["length"] = length.astype(int)
    for name, series in means.items():
        out[name] = series
    for name, series in variances.items():
        out[name] = series

    if not means:
        # Nothing matched the CoverM naming — return the original frame so the
        # caller at least sees its data rather than an empty table.
        return df
    return out


def load_coverage(tsv_path: Path | str) -> pd.DataFrame:
    """Load a CoverM coverage.tsv and normalize it to the internal schema."""
    df = pd.read_csv(tsv_path, sep="\t", index_col=0)
    return normalize_coverage(df)
