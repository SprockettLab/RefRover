"""
MAG-quality benchmark orchestrator: the gold-standard evaluation of selection methods.

For each ``(matrix, rule, k, focal assembly)`` cell, runs the real pipeline —
co-map selection → alignment → CoverM → MetaBAT2 → CheckM2 — and counts
quality-weighted MAGs. Each cell also records the proxy scores (frac_variance,
effective_rank, tiered_axis_count) for the same selection so they can be
correlated with real MAG yield after the run.

**Transpose / co-map design:** binning a focal assembly ``A_f`` needs several
samples' reads over its contig set. "Pick k samples to co-map onto ``A_f``" is
``rule.select(M.T, f)`` — the rules run on the *transposed* matrix with the focal
assembly as the query. The proxy scores run on the same transposition.

All external tools are subprocess-invoked (reusing `align._run`/`_sort_index`,
`coverage.run_coverm`, `formatters.metabat2.write_metabat2`, `binning.run_metabat2`,
`checkm2.run_checkm2`), so the harness is unit-testable with mocked subprocess and
needs the tools only at real run time. Use ``--shard i/N`` (1-based) to split the
focal-assembly list across SLURM array tasks; call ``aggregate_results`` once all
shards finish.
"""

from __future__ import annotations

import json
import subprocess
import time
import traceback
import warnings
from pathlib import Path

import pandas as pd

from refrover.align import _run, _sort_index
from refrover.binning import list_bins, run_metabat2
from refrover.checkm2 import run_checkm2
from refrover.coverage import load_coverage, run_coverm
from refrover.formatters.metabat2 import write_metabat2
from refrover.grid import FeatureSpaceSpec, _instantiate, _rule_valid_for  # re-exported for CLI convenience
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

    Indexes ``focal_fasta`` once, then aligns all k samples concurrently using
    ``threads`` total threads divided among them (1 thread/alignment minimum).
    Each alignment writes ``bams_dir/{sample}.sorted.bam``. Idempotent per BAM.
    """
    import concurrent.futures

    focal_fasta = Path(focal_fasta)
    bams_dir = Path(bams_dir)
    bams_dir.mkdir(parents=True, exist_ok=True)
    reads = manifest.set_index("sample_id")

    sentinel = Path(str(focal_fasta) + _BWA_INDEX_SENTINEL)
    if force or not sentinel.exists():
        _run([aligner_bin, "index", str(focal_fasta)],
             f"{aligner_bin} index {focal_fasta.name}")

    to_align = [s for s in samples
                if force or not (bams_dir / f"{s}.sorted.bam").exists()]
    if not to_align:
        return bams_dir

    # Spread the thread budget evenly across concurrent alignments; at least 1
    # thread per job so BWA-MEM2 doesn't serialize on the GIL.
    threads_per_job = max(1, threads // len(to_align))

    def _align_one(sid: str) -> None:
        bam = bams_dir / f"{sid}.sorted.bam"
        row = reads.loc[sid]
        mem_cmd = [aligner_bin, "mem", "-t", str(threads_per_job),
                   str(focal_fasta), str(row["r1"])]
        r2 = row.get("r2")
        if isinstance(r2, str) and r2:
            mem_cmd.append(r2)
        # Pipe directly into samtools sort — no SAM written to disk.
        sort_cmd = [samtools_bin, "sort", "-@", str(threads_per_job),
                    "-o", str(bam)]
        bwa = subprocess.Popen(mem_cmd, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
        srt = subprocess.Popen(sort_cmd, stdin=bwa.stdout,
                               stderr=subprocess.PIPE)
        bwa.stdout.close()
        _, bwa_err = bwa.communicate()
        _, sort_err = srt.communicate()
        if bwa.returncode != 0:
            raise RuntimeError(
                f"{aligner_bin} mem failed for {sid}:\n{bwa_err.decode()}")
        if srt.returncode != 0:
            raise RuntimeError(
                f"samtools sort failed for {sid}:\n{sort_err.decode()}")
        subprocess.run([samtools_bin, "index", str(bam)],
                       check=True, capture_output=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(to_align)) as pool:
        futures = {pool.submit(_align_one, sid): sid for sid in to_align}
        for fut in concurrent.futures.as_completed(futures):
            fut.result()  # re-raise any alignment exception

    return bams_dir


def run_cell(
    matrix_T: pd.DataFrame,
    matrix_name: str,
    rule_name: str,
    k: int | str,
    focal: str,
    *,
    manifest: pd.DataFrame,
    assemblies: dict[str, Path | str],
    outdir: Path | str,
    threshold: float = 0.0,
    clades: pd.Series | None = None,
    checkm2_db: Path | str | None = None,
    threads: int = 8,
    min_contig: int = 1500,
    aligner_bin: str = "bwa-mem2",
    samtools_bin: str = "samtools",
    metabat2_path: str = "metabat2",
    checkm2_path: str = "checkm2",
    adaptive_k_method: str = "containment_saturation",
    adaptive_k_min: int = 3,
    adaptive_k_max: int = 20,
    force: bool = False,
) -> dict:
    """
    Run one ``(matrix, rule, k, focal)`` cell end-to-end and return a result row.

    Selects co-map samples on ``matrix_T`` (assemblies × samples), aligns them
    onto the focal assembly, computes per-contig coverage, bins with MetaBAT2,
    assesses quality with CheckM2, and counts quality-weighted MAGs. Proxy scores
    (frac_variance, effective_rank, tiered_axis_count) are computed on the same
    selection for later correlation with real MAG yield. Idempotent: an existing
    ``result.json`` is reused unless ``force``.

    When ``k="adaptive"``, the number of co-map samples is estimated per focal
    using :func:`~refrover.adaptive_k.estimate_k` (``adaptive_k_method``).
    The result row records both ``k="adaptive"`` and ``k_actual=<int>``.
    """
    cell_dir = Path(outdir) / matrix_name / f"{rule_name}_k{k}" / focal
    result_json = cell_dir / "result.json"
    if result_json.exists() and not force:
        return json.loads(result_json.read_text())
    cell_dir.mkdir(parents=True, exist_ok=True)

    # Resolve adaptive k: estimate per-focal using the containment saturation curve.
    if k == "adaptive":
        from refrover.adaptive_k import estimate_k
        k_actual = estimate_k(
            matrix_T, focal,
            min_jaccard=threshold,
            method=adaptive_k_method,
            k_min=adaptive_k_min,
            k_max=adaptive_k_max,
        )
    else:
        k_actual = int(k)

    # Build the rule via _instantiate so taxonomy_stratified receives clades.
    _spec = FeatureSpaceSpec(name=matrix_name, matrix=matrix_T.T,
                             threshold=threshold, clades=clades)
    co_map = _instantiate(rule_name, k_actual, _spec).select(matrix_T, focal)
    proxy_scores = score_selection(matrix_T, co_map)

    import shutil

    t0 = time.perf_counter()
    try:
        coverage_dir = cell_dir / "coverage"
        coverage_tsv = coverage_dir / "coverage.tsv"
        if not coverage_tsv.exists() or force:
            bams = align_samples_to_focal(
                assemblies[focal], co_map, manifest, cell_dir / "bams",
                threads=threads, aligner_bin=aligner_bin,
                samtools_bin=samtools_bin, force=force,
            )
            run_coverm(bams, coverage_dir, threads=threads, force=force)
            # Delete BAMs immediately — coverage TSV is all that's needed.
            shutil.rmtree(bams, ignore_errors=True)
        cov_df = load_coverage(coverage_tsv)
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
            "matrix": matrix_name,
            "rule": rule_name,
            "k": k,
            "k_actual": k_actual if k == "adaptive" else None,
            "focal": focal,
            "n_comap": len(co_map),
            **mags,
            **proxy_scores,
            "wall_seconds": wall,
            "cpu_seconds": wall * threads,
            "error": None,
        }
    except Exception as exc:
        wall = time.perf_counter() - t0
        tb = traceback.format_exc()
        error_json = cell_dir / "error.json"
        row = {
            "matrix": matrix_name,
            "rule": rule_name,
            "k": k,
            "k_actual": k_actual if k == "adaptive" else None,
            "focal": focal,
            "n_comap": len(co_map),
            "error": str(exc),
            "traceback": tb,
            "wall_seconds": wall,
        }
        error_json.write_text(json.dumps(row, indent=2))
        warnings.warn(
            f"Cell ({matrix_name}, {rule_name}, k={k}, focal={focal}) failed: {exc}\n"
            f"Details → {error_json}",
            stacklevel=2,
        )
        return row

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
    k_values: list[int | str],
    focals: list[str],
    *,
    manifest: pd.DataFrame,
    assemblies: dict[str, Path | str],
    outdir: Path | str,
    checkm2_db: Path | str | None = None,
    threads: int = 8,
    shard: str | None = None,
    adaptive_k_method: str = "containment_saturation",
    adaptive_k_min: int = 3,
    adaptive_k_max: int = 20,
    **cell_kw,
) -> pd.DataFrame:
    """
    Sweep the benchmark grid: matrices × rules × k × focal assemblies.

    ``feature_spaces`` is a list of :class:`~refrover.grid.FeatureSpaceSpec`
    (each wraps a samples × assemblies matrix, a name, and a candidate threshold).
    Each matrix is transposed for the co-map selection. A cell is run via
    :func:`run_cell` and is resumable. With ``shard='i/N'`` only that round-robin
    slice of focals runs and the combined TSV is *not* written — call
    :func:`aggregate_results` once all shards finish.

    ``k_values`` may include the string ``"adaptive"`` to trigger per-focal k
    estimation via :func:`~refrover.adaptive_k.estimate_k`.

    Returns a tidy DataFrame of all cells this call ran.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    my_focals = _shard(focals, shard)

    rows = []
    for spec in feature_spaces:
        matrix_T = spec.matrix.T
        for rule_name in rules:
            if not _rule_valid_for(rule_name, spec):
                warnings.warn(
                    f"Rule '{rule_name}' skipped for feature space '{spec.name}' "
                    "(taxonomy_stratified requires clades — build with "
                    "'refrover gtdb-gather' + 'refrover gtdb-matrix' first)",
                    stacklevel=2,
                )
                continue
            for k in k_values:
                for focal in my_focals:
                    if focal not in matrix_T.columns:
                        warnings.warn(f"focal '{focal}' absent from {spec.name}; skipping",
                                      stacklevel=2)
                        continue
                    rows.append(run_cell(
                        matrix_T, spec.name, rule_name, k, focal,
                        manifest=manifest, assemblies=assemblies, outdir=outdir,
                        threshold=spec.threshold, clades=spec.clades,
                        checkm2_db=checkm2_db, threads=threads,
                        adaptive_k_method=adaptive_k_method,
                        adaptive_k_min=adaptive_k_min,
                        adaptive_k_max=adaptive_k_max,
                        **cell_kw,
                    ))

    df = pd.DataFrame(rows)
    if shard is None:
        df.to_csv(outdir / "benchmark_results.tsv", sep="\t", index=False)
    return df


def aggregate_results(outdir: Path | str) -> pd.DataFrame:
    """Collect every cell's ``result.json`` (and ``error.json``) under ``outdir``.

    Backfills ``sum_qs`` for cells whose ``result.json`` predates that field by
    reading the sibling ``checkm2/quality_report.tsv`` when present.
    """
    from refrover.mag_quality import count_mags
    from refrover.checkm2 import load_quality_report, QUALITY_REPORT

    outdir = Path(outdir)
    rows = []
    for p in sorted(outdir.rglob("result.json")):
        row = json.loads(p.read_text())
        if "sum_qs" not in row:
            qr = p.parent / "checkm2" / QUALITY_REPORT
            if qr.exists():
                try:
                    quality = load_quality_report(qr)
                    row["sum_qs"] = count_mags(quality)["sum_qs"]
                except Exception:
                    row["sum_qs"] = None
            else:
                row["sum_qs"] = None
        rows.append(row)
    for p in sorted(outdir.rglob("error.json")):
        rows.append(json.loads(p.read_text()))
    return pd.DataFrame(rows)


__all__ = [
    "FeatureSpaceSpec",
    "align_samples_to_focal",
    "run_cell",
    "run_pilot",
    "aggregate_results",
]
