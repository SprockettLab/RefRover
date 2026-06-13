"""
Tier-2 MAG-yield benchmark orchestrator (PLAN.md §8, build sequencing step 2).

Settles the open question Tier-1 could not: *which upstream proxy predicts real
MAG yield?* For each ``(feature_space, rule, k, focal assembly)`` cell it runs the
real pipeline — co-map selection → alignment → CoverM → MetaBAT2 → CheckM2 —
counts quality-weighted MAGs, and records the Tier-1 proxy of the same selection
so the two can be correlated downstream (`analysis.tier_correlation`).

**Transpose / co-map design (locked with the user):** binning a focal assembly
``A_f`` needs several samples' reads over its one contig set. "Pick k samples to
co-map onto ``A_f``" is exactly ``rule.select(M.T, f)`` — the existing rules run
on the *transposed* feature matrix with the focal assembly as the query. So
selection + Tier-1 scoring reuse the Tier-1 stack verbatim on ``M.T``; only the
downstream execution here is new.

All external tools are subprocess-invoked (reusing `align._run`/`_sort_index`,
`coverage.run_coverm`, `formatters.metabat2.write_metabat2`, `binning.run_metabat2`,
`checkm2.run_checkm2`), so the harness is unit-testable with mocked subprocess and
needs the tools only at real run time.
"""

from __future__ import annotations

import json
import subprocess
import time
import warnings
from pathlib import Path

import pandas as pd

from refrover.align import _run, _sort_index
from refrover.binning import list_bins, run_metabat2
from refrover.checkm2 import run_checkm2
from refrover.coverage import load_coverage, run_coverm
from refrover.formatters.metabat2 import write_metabat2
from refrover.mag_quality import count_mags
from refrover.rules import RULE_REGISTRY
from refrover.scores import score_selection

# bwa-mem2 index sentinel (one of the files `bwa-mem2 index` writes).
_BWA_INDEX_SENTINEL = ".bwt.2bit.64"


def align_samples_to_focal(
    focal_fasta: Path | str,
    samples: list[str],
    manifest: pd.DataFrame,
    bams_dir: Path | str,
    *,
    threads: int = 8,
    aligner_bin: str = "bwa-mem2",
    samtools_bin: str = "samtools",
    force: bool = False,
) -> Path:
    """
    Align each co-map sample's reads onto the focal assembly → one sorted BAM each.

    Indexes ``focal_fasta`` once, then for each sample in ``samples`` aligns its
    reads (manifest ``r1``/``r2``) to it, writing ``bams_dir/{sample}.sorted.bam``
    so `coverage.run_coverm` consumes the directory unchanged. Idempotent per BAM.
    """
    focal_fasta = Path(focal_fasta)
    bams_dir = Path(bams_dir)
    bams_dir.mkdir(parents=True, exist_ok=True)
    reads = manifest.set_index("sample_id")

    sentinel = Path(str(focal_fasta) + _BWA_INDEX_SENTINEL)
    if force or not sentinel.exists():
        _run([aligner_bin, "index", str(focal_fasta)],
             f"{aligner_bin} index {focal_fasta.name}")

    for sid in samples:
        bam = bams_dir / f"{sid}.sorted.bam"
        if bam.exists() and not force:
            continue
        row = reads.loc[sid]
        mem = [aligner_bin, "mem", "-t", str(threads), str(focal_fasta), str(row["r1"])]
        r2 = row.get("r2")
        if isinstance(r2, str) and r2:
            mem.append(r2)
        sam = bams_dir / f"{sid}.sam"
        with open(sam, "w") as fh:
            proc = subprocess.run(mem, stdout=fh, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"{aligner_bin} mem failed for {sid}:\n{proc.stderr}")
        _sort_index(sam, bam, samtools_bin, threads)
        sam.unlink(missing_ok=True)

    return bams_dir


def run_cell(
    matrix_T: pd.DataFrame,
    feature_space: str,
    rule_name: str,
    k: int,
    focal: str,
    *,
    manifest: pd.DataFrame,
    assemblies: dict[str, Path | str],
    outdir: Path | str,
    threshold: float = 0.0,
    checkm2_db: Path | str | None = None,
    threads: int = 8,
    min_contig: int = 1500,
    aligner_bin: str = "bwa-mem2",
    samtools_bin: str = "samtools",
    metabat2_path: str = "metabat2",
    checkm2_path: str = "checkm2",
    force: bool = False,
) -> dict:
    """
    Run one ``(feature_space, rule, k, focal)`` cell end-to-end → a result row.

    Selects the co-map sample set on ``matrix_T`` (assemblies × samples), aligns
    those samples onto the focal assembly, computes coverage, bins with MetaBAT2,
    scores bins with CheckM2, and counts quality-weighted MAGs. Also records the
    Tier-1 proxy family of the same co-map selection (``score_selection`` on
    ``matrix_T``) and the compute spent. Idempotent: a finished cell's
    ``result.json`` is reused unless ``force``.
    """
    cell_dir = Path(outdir) / feature_space / f"{rule_name}_k{k}" / focal
    result_json = cell_dir / "result.json"
    if result_json.exists() and not force:
        return json.loads(result_json.read_text())
    cell_dir.mkdir(parents=True, exist_ok=True)

    co_map = RULE_REGISTRY[rule_name](k=k, threshold=threshold).select(matrix_T, focal)
    tier1 = score_selection(matrix_T, co_map)

    t0 = time.perf_counter()
    bams = align_samples_to_focal(
        assemblies[focal], co_map, manifest, cell_dir / "bams",
        threads=threads, aligner_bin=aligner_bin, samtools_bin=samtools_bin, force=force,
    )
    run_coverm(bams, cell_dir / "coverage", threads=threads, force=force)
    cov_df = load_coverage(cell_dir / "coverage" / "coverage.tsv")
    depth_txt = write_metabat2(cov_df, cell_dir, force=force)
    bins_dir = run_metabat2(
        assemblies[focal], depth_txt, cell_dir / "bins",
        min_contig=min_contig, threads=threads, force=force, metabat2_path=metabat2_path,
    )

    if list_bins(bins_dir):
        quality = run_checkm2(
            bins_dir, cell_dir / "checkm2", db_path=checkm2_db,
            threads=threads, force=force, checkm2_path=checkm2_path,
        )
    else:
        quality = pd.DataFrame({"Completeness": [], "Contamination": []})
    mags = count_mags(quality)
    wall = time.perf_counter() - t0

    row = {
        "feature_space": feature_space,
        "rule": rule_name,
        "k": k,
        "focal": focal,
        "n_comap": len(co_map),
        **mags,
        **tier1,
        "wall_seconds": wall,
        "cpu_seconds": wall * threads,
    }
    result_json.write_text(json.dumps(row))
    return row


def _shard(items: list, shard: str | None) -> list:
    """Round-robin shard 'i/N' (1-based) of a list; None → all."""
    if shard is None:
        return items
    i, n = (int(x) for x in shard.split("/"))
    if not (1 <= i <= n):
        raise ValueError(f"--shard i/N requires 1 <= i <= N, got {shard}")
    return items[i - 1 :: n]


def run_pilot(
    feature_spaces: list,
    rules: list[str],
    k_values: list[int],
    focals: list[str],
    *,
    manifest: pd.DataFrame,
    assemblies: dict[str, Path | str],
    outdir: Path | str,
    checkm2_db: Path | str | None = None,
    threads: int = 8,
    shard: str | None = None,
    **cell_kw,
) -> pd.DataFrame:
    """
    Sweep the Tier-2 grid: feature_spaces × rules × k × focal assemblies.

    ``feature_spaces`` is a list of `grid.FeatureSpaceSpec` (the samples ×
    assemblies matrices); each is transposed once for the co-map selection. A cell
    is run via :func:`run_cell` (resumable). With ``shard='i/N'`` only that
    round-robin slice of focals runs and the combined TSV is *not* written —
    call :func:`aggregate_results` once all shards finish.

    Returns the tidy results frame for the focals this call ran.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    my_focals = _shard(focals, shard)

    rows = []
    for spec in feature_spaces:
        matrix_T = spec.matrix.T
        for rule_name in rules:
            for k in k_values:
                for focal in my_focals:
                    if focal not in matrix_T.columns:
                        warnings.warn(f"focal '{focal}' absent from {spec.name}; skipping",
                                      stacklevel=2)
                        continue
                    rows.append(run_cell(
                        matrix_T, spec.name, rule_name, k, focal,
                        manifest=manifest, assemblies=assemblies, outdir=outdir,
                        threshold=spec.threshold, checkm2_db=checkm2_db,
                        threads=threads, **cell_kw,
                    ))

    df = pd.DataFrame(rows)
    if shard is None:
        df.to_csv(outdir / "tier2_results.tsv", sep="\t", index=False)
    return df


def aggregate_results(outdir: Path | str) -> pd.DataFrame:
    """Collect every cell's ``result.json`` under ``outdir`` into one tidy TSV."""
    outdir = Path(outdir)
    rows = [json.loads(p.read_text()) for p in sorted(outdir.rglob("result.json"))]
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(outdir / "tier2_results.tsv", sep="\t", index=False)
    return df


__all__ = ["align_samples_to_focal", "run_cell", "run_pilot", "aggregate_results"]
