"""
GTDB-mediated feature space (PLAN.md §3, feature space 3).

Runs ``sourmash gather`` on each sample's read sketches against a GTDB
signature database, aggregates per-sample CSVs into a ``samples × taxa``
abundance matrix, and derives per-assembly clade labels for
``TaxonomyStratifiedRule``.

The gather step is embarrassingly parallel — use ``refrover gtdb-gather
--shard i/N`` for SLURM array jobs, then ``refrover gtdb-matrix`` once
all shards finish. The full matrix is then passed to ``refrover benchmark
--gtdb-matrix`` (or found automatically when ``taxonomy_stratified`` is
requested and the default cache path exists).

GTDB sourmash databases:
    https://sourmash.readthedocs.io/en/latest/databases.html
    Recommended: GTDB rs214 k=31 scaled=1000  (~3 GB .zip)
    Download with:
        curl -L https://osf.io/th2my/download -o gtdb-rs214-reps.k31.zip
"""

from __future__ import annotations

import subprocess
import warnings
from pathlib import Path

import pandas as pd


# ── per-sample gather ─────────────────────────────────────────────────────────

def run_gather_sample(
    sample_id: str,
    read_sig: Path | str,
    gtdb_db: Path | str,
    outdir: Path | str,
    *,
    ksize: int = 31,
    threshold_bp: int = 50_000,
    force: bool = False,
    sourmash_path: str = "sourmash",
) -> Path | None:
    """
    Run ``sourmash gather`` for one sample's reads against the GTDB database.

    Parameters
    ----------
    sample_id : str
        Used to name the output CSV (``{outdir}/{sample_id}_gather.csv``).
    read_sig : path
        The merged read .sig for this sample (from ``refrover sketch-reads``).
    gtdb_db : path
        GTDB sourmash signature database (.zip).  Download the representative-
        genome database (``gtdb-rs*-reps.k31.zip``) from the sourmash databases
        page; the full-genome database works too but is ~10× larger.
    outdir : path
        Directory to write ``{sample_id}_gather.csv`` into.
    ksize : int
        k-mer size; must match the read sketches and GTDB database (default 31).
    threshold_bp : int
        Minimum k-mer overlap to report a GTDB genome (default 50 kbp).
    force : bool
        Re-run even if the output CSV already exists.

    Returns
    -------
    Path or None
        Path to the gather CSV, or None when sourmash finds no GTDB matches
        (sparse / low-complexity samples).
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / f"{sample_id}_gather.csv"

    if out_csv.exists() and not force:
        return out_csv

    cmd = [
        sourmash_path, "gather",
        str(read_sig),
        str(gtdb_db),
        "--output", str(out_csv),
        "--ksize", str(ksize),
        "--threshold-bp", str(threshold_bp),
        "--no-save-matches",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr_lc = result.stderr.lower()
        if "no matches" in stderr_lc or "found 0" in stderr_lc:
            warnings.warn(
                f"No GTDB matches for '{sample_id}' — sample may be low-complexity "
                "or reads are too divergent from GTDB representatives.",
                UserWarning,
                stacklevel=2,
            )
            return None
        raise RuntimeError(
            f"sourmash gather failed for '{sample_id}':\n{result.stderr}"
        )

    if not out_csv.exists():
        warnings.warn(
            f"sourmash gather for '{sample_id}' exited 0 but wrote no CSV.",
            UserWarning,
            stacklevel=2,
        )
        return None
    return out_csv


# ── aggregate gather CSVs → samples × taxa matrix ────────────────────────────

def aggregate_gather(
    gather_dir: Path | str,
    sample_ids: list[str] | None = None,
    *,
    abund_col: str = "f_unique_to_query",
    name_col: str = "name",
) -> pd.DataFrame:
    """
    Aggregate per-sample gather CSVs into a ``samples × taxa`` abundance matrix.

    Each gather CSV has one row per GTDB genome match. ``f_unique_to_query``
    (the fraction of sample reads uniquely assigned to that genome) is used as
    the abundance value. The taxon name is taken from the ``name`` column (full
    GTDB lineage string as written by sourmash gather).

    Parameters
    ----------
    gather_dir : path
        Directory of ``{sample_id}_gather.csv`` files.
    sample_ids : list of str, optional
        Expected sample IDs. Samples with no gather CSV get all-zero rows;
        a warning is issued per missing sample.

    Returns
    -------
    pd.DataFrame
        ``samples × GTDB-taxon`` matrix; values = ``f_unique_to_query`` in [0,1].
        Index name ``"sample_id"``, columns name ``"taxon"``.
    """
    gather_dir = Path(gather_dir)
    rows: dict[str, dict] = {}

    for csv in sorted(gather_dir.glob("*_gather.csv")):
        sid = csv.stem[: -len("_gather")]
        try:
            df = pd.read_csv(csv)
        except Exception as exc:
            warnings.warn(f"Could not parse {csv.name}: {exc}", stacklevel=2)
            continue
        if name_col not in df.columns or abund_col not in df.columns:
            warnings.warn(
                f"{csv.name} missing '{name_col}' or '{abund_col}' columns; skipping.",
                stacklevel=2,
            )
            continue
        rows[sid] = df.set_index(name_col)[abund_col].to_dict()

    if not rows:
        raise RuntimeError(
            f"No valid gather CSVs found in {gather_dir}.\n"
            "Run 'refrover gtdb-gather' first."
        )

    matrix = pd.DataFrame(rows).T.fillna(0.0)
    matrix.index.name = "sample_id"
    matrix.columns.name = "taxon"

    if sample_ids is not None:
        missing = [s for s in sample_ids if s not in matrix.index]
        if missing:
            warnings.warn(
                f"{len(missing)} sample(s) have no gather results and will be "
                f"all-zero rows: {missing[:5]}{'...' if len(missing) > 5 else ''}",
                stacklevel=2,
            )
        matrix = matrix.reindex(sample_ids, fill_value=0.0)

    return matrix


# ── clade inference from GTDB matrix ─────────────────────────────────────────

_RANK_PREFIX = {
    "domain":  "d__", "phylum": "p__", "class":  "c__",
    "order":   "o__", "family": "f__", "genus":  "g__", "species": "s__",
}


def _extract_rank(taxon_name: str, prefix: str) -> str:
    """Pull the field that starts with *prefix* from a GTDB lineage string."""
    for part in taxon_name.replace(";", " ").split():
        if part.startswith(prefix):
            return part
    return "Unclassified"


def infer_clades(
    gtdb_matrix: pd.DataFrame,
    *,
    level: str = "genus",
) -> pd.Series:
    """
    Derive a clade label per assembly from the GTDB abundance matrix.

    For each sample/assembly row the dominant GTDB taxon (argmax) names the
    clade; the requested taxonomic level is extracted from the GTDB lineage
    string.  Because sample ↔ assembly is 1:1 in RefRover (PLAN §2), the row
    index of *gtdb_matrix* doubles as assembly IDs.

    Parameters
    ----------
    gtdb_matrix : pd.DataFrame
        ``samples × taxa`` matrix from :func:`aggregate_gather` or
        :func:`load_gtdb_matrix`.
    level : str
        Taxonomic rank: ``"domain"``, ``"phylum"``, ``"class"``, ``"order"``,
        ``"family"``, ``"genus"`` (default), or ``"species"``.  Can also be a
        raw GTDB prefix (``"g__"``).

    Returns
    -------
    pd.Series
        Index = sample_id (= assembly_id in 1:1 design), values = clade label
        at the requested level.  All-zero rows (no GTDB matches) get
        ``"Unclassified"``.
    """
    prefix = _RANK_PREFIX.get(level, level if "__" in level else level + "__")

    # All-zero rows have no dominant taxon; argmax returns a valid column name
    # but it's meaningless — detect and replace.
    row_max = gtdb_matrix.max(axis=1)
    dominant = gtdb_matrix.idxmax(axis=1)
    clades = dominant.apply(lambda name: _extract_rank(name, prefix))
    clades[row_max == 0] = "Unclassified"
    clades.name = "clade"
    return clades


# ── I/O helpers ───────────────────────────────────────────────────────────────

def write_gtdb_matrix(matrix: pd.DataFrame, path: Path | str) -> Path:
    """Write a GTDB abundance matrix to a tab-separated file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(path, sep="\t")
    return path


def load_gtdb_matrix(path: Path | str) -> pd.DataFrame:
    """Load a GTDB abundance matrix TSV written by :func:`write_gtdb_matrix`."""
    df = pd.read_csv(path, sep="\t", index_col=0)
    df.index.name = "sample_id"
    df.columns.name = "taxon"
    return df


__all__ = [
    "run_gather_sample",
    "aggregate_gather",
    "infer_clades",
    "write_gtdb_matrix",
    "load_gtdb_matrix",
]
