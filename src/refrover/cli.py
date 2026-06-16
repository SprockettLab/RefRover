"""
RefRover CLI entry point.

Subcommands:
  sketch          Sketch assemblies with sourmash (per-sample, SLURM-shardable)
  sketch-reads    Sketch reads per sample for containment computation (SLURM-shardable)
  jaccard         Compute the assembly-vs-assembly Jaccard similarity matrix
  containment     Compute the reads-vs-assembly containment matrix
  select          Select which assemblies to map against, per sample
  align           Align reads to selected assemblies
  coverage        Compute per-contig depth with CoverM
  format          Format coverage tables for downstream binners
  benchmark       Run the full pipeline (align→bin→CheckM2) across selection methods
                  (also records proxy scores for each selection to enable correlation)
  rank-selectors  Fast alignment-free proxy scoring across selection methods
  run             Full RefRover pipeline end-to-end (single selector)
"""

import click
from pathlib import Path


def _shard(items: list, shard: str | None) -> list:
    """Round-robin shard 'i/N' (1-based) from a list; None → all items."""
    if shard is None:
        return items
    i, n = (int(x) for x in shard.split("/"))
    if not (1 <= i <= n):
        raise click.BadParameter(f"--shard requires 1 <= i <= N, got {shard!r}")
    return items[i - 1 :: n]


@click.group()
@click.version_option()
def main():
    """RefRover: differential coverage tables for metagenomic binners."""


# ── sketch ────────────────────────────────────────────────────────────────────

@main.command()
@click.option("--manifest", required=True, type=click.Path(exists=True),
              help="Sample manifest TSV")
@click.option("--outdir", required=True, type=click.Path(),
              help="Output directory for .sig files")
@click.option("--ksize", default=31, show_default=True)
@click.option("--scaled", default=1000, show_default=True)
@click.option("--threads", default=1, show_default=True)
@click.option("--shard", default=None,
              help="i/N — process only 1-based shard i of N samples (for SLURM arrays)")
@click.option("--force", is_flag=True, help="Re-sketch even if output exists")
def sketch(manifest, outdir, ksize, scaled, threads, shard, force):
    """Sketch assemblies with sourmash sketch dna (one .sig per sample)."""
    from refrover.io import read_manifest
    from refrover.sketch import sketch_assemblies

    df = read_manifest(manifest)
    samples = _shard(df["sample_id"].tolist(), shard)
    fasta_paths = df.set_index("sample_id").loc[samples, "assembly"].tolist()

    sig_paths = sketch_assemblies(
        fasta_paths,
        outdir=outdir,
        ksize=ksize,
        scaled=scaled,
        threads=threads,
        force=force,
    )
    click.echo(f"Sketched {len(sig_paths)} assemblies → {outdir}")


# ── sketch-reads ──────────────────────────────────────────────────────────────

@main.command(name="sketch-reads")
@click.option("--manifest", required=True, type=click.Path(exists=True),
              help="Sample manifest TSV (columns: sample_id, r1, r2 [opt], "
                   "long_reads [opt])")
@click.option("--outdir", required=True, type=click.Path(),
              help="Output directory for read .sig files (one per sample)")
@click.option("--ksize", default=31, show_default=True,
              help="k-mer size; must match the assembly sketches for containment")
@click.option("--scaled", default=1000, show_default=True,
              help="Scaled factor for read MinHash sketches")
@click.option("--shard", default=None,
              help="i/N — process only 1-based shard i of N samples (for SLURM arrays)")
@click.option("--force", is_flag=True, help="Re-sketch even if output exists")
def sketch_reads_cmd(manifest, outdir, ksize, scaled, shard, force):
    """Sketch reads per sample into a merged .sig for containment computation.

    r1, r2, and long_reads (if present) are merged into a single per-sample
    sketch so containment(reads_i, assembly_j) reflects all reads from that
    sample. Use --shard i/N to process one slice of samples per SLURM task.
    """
    from refrover.io import read_manifest
    from refrover.sketch import sketch_reads

    df = read_manifest(manifest)
    rows = df.set_index("sample_id")
    sample_ids = _shard(df["sample_id"].tolist(), shard)

    n_done = 0
    for sid in sample_ids:
        row = rows.loc[sid]
        reads = [
            row.get("r1", ""),
            row.get("r2", ""),
            row.get("long_reads", ""),
        ]
        sketch_reads(sid, reads, outdir=outdir, ksize=ksize, scaled=scaled, force=force)
        n_done += 1

    click.echo(f"Sketched reads for {n_done} sample(s) → {outdir}")


# ── jaccard ───────────────────────────────────────────────────────────────────

@main.command()
@click.option("--assembly-sketches", required=True, type=click.Path(exists=True),
              help="Directory of assembly .sig files (stem = sample_id)")
@click.option("--ksize", default=31, show_default=True)
@click.option("--outdir", required=True, type=click.Path(),
              help="Directory to write jaccard_matrix.tsv")
@click.option("--force", is_flag=True)
def jaccard(assembly_sketches, ksize, outdir, force):
    """Compute the assembly-vs-assembly Jaccard similarity matrix.

    Reads all .sig files from --assembly-sketches (file stem = sample_id) and
    produces a labeled TSV of pairwise Jaccard similarities. This matrix is the
    cheapest input for selection methods; use --containment-matrix for the
    read-informed equivalent.
    """
    from refrover.containment import sigs_by_sample
    from refrover.feature_spaces import jaccard_matrix

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_tsv = outdir / "jaccard_matrix.tsv"

    if out_tsv.exists() and not force:
        click.echo(f"Jaccard matrix already exists at {out_tsv} (use --force to redo)")
        return

    asm_sigs = sigs_by_sample(assembly_sketches)
    if not asm_sigs:
        raise click.ClickException(f"No .sig files found in {assembly_sketches}")

    click.echo(f"Computing Jaccard matrix for {len(asm_sigs)} assemblies…")
    mat = jaccard_matrix(asm_sigs, ksize=ksize)
    mat.to_csv(out_tsv, sep="\t")
    click.echo(f"Jaccard matrix ({mat.shape[0]}×{mat.shape[1]}) → {out_tsv}")


# ── containment ───────────────────────────────────────────────────────────────

@main.command()
@click.option("--read-sketches", required=True, type=click.Path(exists=True),
              help="Directory of read .sig files (one per sample, stem = sample_id)")
@click.option("--assembly-sketches", required=True, type=click.Path(exists=True),
              help="Directory of assembly .sig files (one per sample, stem = sample_id)")
@click.option("--ksize", default=31, show_default=True)
@click.option("--outdir", required=True, type=click.Path())
@click.option("--force", is_flag=True)
def containment(read_sketches, assembly_sketches, ksize, outdir, force):
    """Compute the cross-sample reads-vs-assembly containment matrix."""
    from refrover.containment import (
        compute_containment_matrix, sigs_by_sample, write_containment_matrix,
    )

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_tsv = outdir / "containment_matrix.tsv"

    if out_tsv.exists() and not force:
        click.echo(f"Containment matrix already exists at {out_tsv} (use --force to redo)")
        return

    read_sigs = sigs_by_sample(read_sketches)
    asm_sigs = sigs_by_sample(assembly_sketches)
    if not read_sigs:
        raise click.ClickException(f"No .sig files found in {read_sketches}")
    if not asm_sigs:
        raise click.ClickException(f"No .sig files found in {assembly_sketches}")

    click.echo(f"Computing containment: {len(read_sigs)} read × {len(asm_sigs)} "
               f"assembly sketches…")
    matrix = compute_containment_matrix(read_sigs, asm_sigs, ksize=ksize)
    write_containment_matrix(matrix, out_tsv)
    click.echo(f"Containment matrix ({matrix.shape[0]}×{matrix.shape[1]}) → {out_tsv}")


# ── select ────────────────────────────────────────────────────────────────────

@main.command()
@click.option("--sketches", type=click.Path(exists=True),
              help="Directory of assembly .sig files (Jaccard selectors)")
@click.option("--jaccard-matrix", type=click.Path(exists=True),
              help="Precomputed Jaccard matrix TSV (alternative to --sketches)")
@click.option("--containment-matrix", type=click.Path(exists=True),
              help="Containment matrix TSV (for rules that use read signal)")
@click.option("--manifest", required=True, type=click.Path(exists=True))
@click.option("--selector", default="archetype", show_default=True,
              type=click.Choice(["random", "maxmin", "kmedoids", "archetype",
                                 "greedy_var", "containment"]))
@click.option("--k", default="5", show_default=True,
              help="References per sample, or 'auto' to infer from similarity structure")
@click.option("--adaptive-k-method", default="similarity_gap", show_default=True,
              type=click.Choice(["scree_elbow", "similarity_gap", "saturation_curve",
                                 "containment_saturation"]))
@click.option("--min-jaccard", default=0.1, show_default=True,
              help="Minimum similarity/containment floor for candidate assemblies")
@click.option("--outdir", required=True, type=click.Path())
@click.option("--force", is_flag=True)
def select(sketches, jaccard_matrix, containment_matrix, manifest, selector, k,
           adaptive_k_method, min_jaccard, outdir, force):
    """Select which assemblies to map against, per sample."""
    import pandas as pd
    from refrover.io import read_manifest, write_assignments
    from refrover.selectors import SELECTOR_REGISTRY, CONTAINMENT_SELECTORS
    from refrover.similarity import matrix_from_sigs, load_containment_matrix

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    assignments_path = outdir / "assignments.tsv"

    if assignments_path.exists() and not force:
        click.echo(f"Assignments already exist at {assignments_path} (use --force to redo)")
        return

    df = read_manifest(manifest)

    if selector in CONTAINMENT_SELECTORS:
        if not containment_matrix:
            raise click.ClickException(
                f"--selector {selector} requires --containment-matrix"
            )
        sim_matrix = load_containment_matrix(containment_matrix)
    elif jaccard_matrix:
        sim_matrix = pd.read_csv(jaccard_matrix, sep="\t", index_col=0)
    elif sketches:
        sig_paths = sorted(Path(sketches).glob("*.sig"))
        if not sig_paths:
            raise click.ClickException(f"No .sig files found in {sketches}")
        click.echo(f"Computing similarity matrix from {len(sig_paths)} sketches…")
        sim_matrix = matrix_from_sigs(sig_paths)
    else:
        raise click.ClickException(
            "Provide one of: --jaccard-matrix, --sketches, or --containment-matrix"
        )

    use_adaptive_k = (str(k).lower() == "auto")
    fixed_k = None if use_adaptive_k else int(k)
    sel_cls = SELECTOR_REGISTRY[selector]

    rows = []
    for _, row in df.iterrows():
        sid = row["sample_id"]
        if use_adaptive_k:
            from refrover.adaptive_k import estimate_k
            k_i = estimate_k(sim_matrix, sid, min_jaccard=min_jaccard,
                             method=adaptive_k_method)
        else:
            k_i = fixed_k

        sel = sel_cls(k=k_i, min_jaccard=min_jaccard)
        prototypes = sel.select(sim_matrix, sid)
        rows.append({
            "sample_id": sid,
            "prototype_ids": ",".join(prototypes),
            "n_prototypes": len(prototypes),
            "k_estimated": k_i if use_adaptive_k else None,
        })

    assignments = pd.DataFrame(rows)
    write_assignments(assignments, assignments_path)
    click.echo(f"Assignments → {assignments_path}")


# ── align ─────────────────────────────────────────────────────────────────────

@main.command()
@click.option("--assignments", required=True, type=click.Path(exists=True))
@click.option("--manifest", required=True, type=click.Path(exists=True))
@click.option("--outdir", required=True, type=click.Path())
@click.option("--aligner", default="bwa-mem2", show_default=True,
              type=click.Choice(["bwa-mem2", "bwa", "minimap2"]))
@click.option("--threads", default=8, show_default=True)
@click.option("--force", is_flag=True)
def align(assignments, manifest, outdir, aligner, threads, force):
    """Align reads to selected assemblies."""
    from refrover.align import run_alignment
    from refrover.io import read_manifest
    import pandas as pd

    df = read_manifest(manifest)
    asgn = pd.read_csv(assignments, sep="\t")
    bam_paths = run_alignment(df, asgn, outdir=outdir, aligner=aligner,
                              threads=threads, force=force)
    click.echo(f"Aligned {len(bam_paths)} samples → {outdir}")


# ── coverage ──────────────────────────────────────────────────────────────────

@main.command()
@click.option("--bams", required=True, type=click.Path(exists=True),
              help="Directory of sorted BAM files")
@click.option("--outdir", required=True, type=click.Path())
@click.option("--threads", default=8, show_default=True)
@click.option("--force", is_flag=True)
def coverage(bams, outdir, threads, force):
    """Compute per-contig depth with CoverM."""
    from refrover.coverage import run_coverm
    run_coverm(bams_dir=bams, outdir=outdir, threads=threads, force=force)
    click.echo(f"Coverage → {outdir}")


# ── format ────────────────────────────────────────────────────────────────────

@main.command()
@click.option("--coverage", "coverage_dir", required=True, type=click.Path(exists=True))
@click.option("--binners", default="generic", show_default=True,
              help="Comma-separated list: metabat2,semibin2,maxbin2,concoct,generic")
@click.option("--outdir", required=True, type=click.Path())
@click.option("--force", is_flag=True)
def format(coverage_dir, binners, outdir, force):
    """Format coverage tables for downstream binners."""
    from refrover.formatters import format_for_binner
    from refrover.coverage import load_coverage

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    binner_list = [b.strip() for b in binners.split(",")]
    cov_tsv = Path(coverage_dir) / "coverage.tsv"
    df = load_coverage(cov_tsv)

    for binner in binner_list:
        out = format_for_binner(df, binner=binner, outdir=outdir, force=force)
        click.echo(f"{binner} → {out}")


# ── benchmark ─────────────────────────────────────────────────────────────────

@main.command()
@click.option("--manifest", required=True, type=click.Path(exists=True),
              help="Sample manifest TSV (sample_id, assembly, r1, r2 [opt], "
                   "long_reads [opt])")
@click.option("--assemblies-dir", required=True, type=click.Path(exists=True),
              help="Directory of per-sample assembly FASTAs (stem = sample_id)")
@click.option("--jaccard-matrix", type=click.Path(exists=True),
              help="Precomputed Jaccard similarity matrix TSV "
                   "(from refrover jaccard)")
@click.option("--containment-matrix", type=click.Path(exists=True),
              help="Precomputed containment matrix TSV "
                   "(from refrover containment)")
@click.option("--gtdb-matrix", type=click.Path(),
              help="Precomputed GTDB abundance matrix TSV "
                   "(from refrover gtdb-matrix; derived clade labels are attached "
                   "to all feature spaces to enable taxonomy_stratified)")
@click.option("--gtdb-db", default=None, type=click.Path(exists=True),
              help="GTDB sourmash database (.zip) for auto-building the matrix "
                   "when --gtdb-matrix is absent and taxonomy_stratified is requested. "
                   "Requires --read-sketches. Not available with --shard.")
@click.option("--read-sketches", default=None, type=click.Path(exists=True),
              help="Directory of per-sample read .sig files; used with --gtdb-db "
                   "for auto-building the GTDB matrix.")
@click.option("--gtdb-clade-level", default="genus", show_default=True,
              help="Taxonomic level for clade labels derived from the GTDB matrix "
                   "(domain, phylum, class, order, family, genus, species)")
@click.option("--rules", default="random,maxmin,css", show_default=True,
              help="Comma-separated selection rules to benchmark")
@click.option("--k-range", default="3,5,8,10", show_default=True,
              help="Comma-separated k values to sweep; include 'adaptive' to let "
                   "RefRover pick k per focal from the marginal-variance saturation curve "
                   "(e.g. '3,5,8,10,adaptive')")
@click.option("--adaptive-k-method",
              default="containment_saturation", show_default=True,
              type=click.Choice(["containment_saturation", "scree_elbow",
                                 "similarity_gap", "saturation_curve"]),
              help="Estimator used when k='adaptive'. "
                   "'containment_saturation' (default) is the validated method: greedy "
                   "residual-variance forward selection, stopping when marginal gain "
                   "falls below 10%% of the first gain.")
@click.option("--focals", default=None, type=click.Path(exists=True),
              help="File listing focal sample_ids to benchmark (one per line); "
                   "default is all samples in the manifest")
@click.option("--shard", default=None,
              help="i/N — run only 1-based shard i of N focal assemblies "
                   "(for SLURM array jobs; call aggregate-results when all done)")
@click.option("--checkm2-db", default=None, type=click.Path(exists=True),
              help="CheckM2 DIAMOND database path "
                   "(~/checkm2_db/CheckM2_database/uniref100.KO.1.dmnd)")
@click.option("--checkm2-path", default="checkm2", show_default=True,
              help="Path to the checkm2 binary; use this when checkm2 lives in "
                   "a separate conda env, e.g. "
                   "$(conda run -n checkm2 which checkm2)")
@click.option("--min-similarity", default=0.0, show_default=True,
              help="Minimum similarity/containment floor for co-map candidates")
@click.option("--min-contig", default=1500, show_default=True,
              help="MetaBAT2 minimum contig length (-m)")
@click.option("--threads", default=8, show_default=True)
@click.option("--aligner", default="bwa-mem2", show_default=True,
              type=click.Choice(["bwa-mem2", "bwa"]))
@click.option("--outdir", required=True, type=click.Path(),
              help="Output directory; each cell lands in "
                   "outdir/{matrix}/{rule}_k{k}/{focal}/")
@click.option("--force", is_flag=True,
              help="Re-run cells that already have a result.json")
def benchmark(manifest, assemblies_dir, jaccard_matrix, containment_matrix,
              gtdb_matrix, gtdb_db, read_sketches, gtdb_clade_level,
              rules, k_range, adaptive_k_method, focals, shard, checkm2_db, checkm2_path,
              min_similarity, min_contig, threads, aligner, outdir, force):
    """Benchmark selection methods by running the full pipeline and measuring MAG quality.

    For each combination of (input matrix) × (rule) × (k) × (focal assembly),
    selects co-map samples, aligns their reads to the focal assembly, bins
    contigs with MetaBAT2, and assesses bin quality with CheckM2. Proxy scores
    (frac_variance, effective_rank, tiered_axis_count) are recorded alongside
    real MAG yield so they can be correlated after the run.

    Use --shard i/N to split focal assemblies across SLURM array tasks, then
    run 'refrover aggregate-results --outdir ...' to merge all shards.
    """
    import pandas as pd
    from refrover.io import read_manifest
    from refrover.mag_benchmark import run_pilot, FeatureSpaceSpec

    df = read_manifest(manifest)
    rule_list = [r.strip() for r in rules.split(",")]

    # ── resolve GTDB matrix when taxonomy_stratified is requested ─────────────
    # The GTDB matrix is NOT a separate feature space for Tier-2 selection
    # (taxa are not sample-dimension vectors). Instead, it is used to derive
    # per-assembly clade labels which are attached to all feature spaces.
    gtdb_mat_path = gtdb_matrix
    if "taxonomy_stratified" in rule_list and not gtdb_mat_path:
        # 1. Check conventional cache location: {outdir}/../gtdb/gtdb_matrix.tsv
        cached = Path(outdir).parent / "gtdb" / "gtdb_matrix.tsv"
        if cached.exists():
            gtdb_mat_path = str(cached)
            click.echo(f"Using cached GTDB matrix: {cached}")
        elif gtdb_db and read_sketches:
            if shard:
                raise click.ClickException(
                    "Auto-building the GTDB matrix is not available in --shard mode.\n"
                    "Run 'refrover gtdb-gather --shard i/N' then 'refrover gtdb-matrix' "
                    "first, and pass the result with --gtdb-matrix."
                )
            click.echo(
                "taxonomy_stratified requested but --gtdb-matrix not provided.\n"
                f"Auto-building from {read_sketches} × {gtdb_db} …\n"
                "(this may take several minutes for large datasets)"
            )
            from refrover.gtdb import (
                run_gather_sample, aggregate_gather, write_gtdb_matrix,
            )
            gather_dir = Path(outdir).parent / "gtdb" / "gather"
            gather_dir.mkdir(parents=True, exist_ok=True)
            sig_dir = Path(read_sketches)
            for sid in df["sample_id"]:
                sig = sig_dir / f"{sid}.sig"
                if not sig.exists():
                    click.echo(f"  WARNING: read sig missing for {sid}, skipped", err=True)
                    continue
                run_gather_sample(sid, sig, gtdb_db, gather_dir, force=force)
            gtdb_matrix_df = aggregate_gather(gather_dir, df["sample_id"].tolist())
            gtdb_mat_path = str(cached)
            write_gtdb_matrix(gtdb_matrix_df, gtdb_mat_path)
            click.echo(f"GTDB matrix ({gtdb_matrix_df.shape[0]}×{gtdb_matrix_df.shape[1]}) "
                       f"→ {gtdb_mat_path}")
        else:
            click.echo(
                "WARNING: taxonomy_stratified requested but no GTDB matrix available.\n"
                "  Options:\n"
                "    --gtdb-matrix <path>        prebuilt matrix\n"
                "    --gtdb-db <db> --read-sketches <dir>  auto-build (non-sharded only)\n"
                "taxonomy_stratified will be skipped for all feature spaces.",
                err=True,
            )

    # Derive clade labels from the GTDB matrix (if available).
    clades = None
    if gtdb_mat_path and Path(gtdb_mat_path).exists():
        from refrover.gtdb import infer_clades, load_gtdb_matrix
        _gtdb_df = load_gtdb_matrix(gtdb_mat_path)
        clades = infer_clades(_gtdb_df, level=gtdb_clade_level)
        click.echo(
            f"Loaded GTDB clades ({clades.nunique()} unique {gtdb_clade_level}s) "
            f"from {gtdb_mat_path}"
        )

    # Build the list of input matrices to sweep.
    matrix_specs = []
    if jaccard_matrix:
        mat = pd.read_csv(jaccard_matrix, sep="\t", index_col=0)
        matrix_specs.append(FeatureSpaceSpec(name="jaccard", matrix=mat,
                                             threshold=min_similarity, clades=clades))
    if containment_matrix:
        mat = pd.read_csv(containment_matrix, sep="\t", index_col=0)
        matrix_specs.append(FeatureSpaceSpec(name="containment", matrix=mat,
                                             threshold=min_similarity, clades=clades))

    if not matrix_specs:
        raise click.ClickException(
            "Provide at least one of: --jaccard-matrix, --containment-matrix"
        )

    k_list = [
        k.strip() if k.strip().lower() == "adaptive" else int(k.strip())
        for k in k_range.split(",")
    ]

    # Focal assemblies to evaluate.
    if focals:
        focal_ids = [line.strip() for line in Path(focals).read_text().splitlines()
                     if line.strip()]
    else:
        focal_ids = df["sample_id"].tolist()

    # Build sample_id → assembly FASTA path from assemblies-dir.
    asm_dir = Path(assemblies_dir)
    assemblies: dict[str, Path] = {}
    for sid in focal_ids:
        # Match stem = sample_id with any of the common FASTA extensions.
        for ext in (".fasta", ".fa", ".fna"):
            p = asm_dir / f"{sid}{ext}"
            if p.exists():
                assemblies[sid] = p
                break
        else:
            # Try manifest assembly column as fallback.
            row = df.set_index("sample_id").loc[sid] if sid in df["sample_id"].values else None
            if row is not None and Path(row["assembly"]).exists():
                assemblies[sid] = Path(row["assembly"])

    missing = [sid for sid in focal_ids if sid not in assemblies]
    if missing:
        raise click.ClickException(
            f"{len(missing)} focal assembly FASTA(s) not found in {assemblies_dir}: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )

    click.echo(
        f"Benchmark: {len(matrix_specs)} matrix(ces) × {len(rule_list)} rules × "
        f"{len(k_list)} k values × {len(focal_ids)} focal assemblies"
    )
    if shard:
        click.echo(f"  Running shard {shard}")

    results = run_pilot(
        feature_spaces=matrix_specs,
        rules=rule_list,
        k_values=k_list,
        focals=focal_ids,
        manifest=df,
        assemblies=assemblies,
        outdir=outdir,
        checkm2_db=checkm2_db,
        checkm2_path=checkm2_path,
        threads=threads,
        shard=shard,
        min_contig=min_contig,
        aligner_bin=aligner,
        adaptive_k_method=adaptive_k_method,
        force=force,
    )

    if not results.empty:
        click.echo(f"\n{len(results)} cells complete. Top results by weighted MAGs:")
        top = (results.sort_values("weighted_mags", ascending=False)
               .head(10)[["matrix", "rule", "k", "focal", "weighted_mags",
                           "frac_variance"]]
               .to_string(index=False))
        click.echo(top)
    click.echo(f"\nResults → {outdir}")


# ── aggregate-results ─────────────────────────────────────────────────────────

@main.command(name="aggregate-results")
@click.option("--outdir", required=True, type=click.Path(exists=True),
              help="The --outdir used for refrover benchmark (contains cell result.json files)")
@click.option("--out-tsv", default=None, type=click.Path(),
              help="Output TSV path (default: outdir/benchmark_results.tsv)")
def aggregate_results_cmd(outdir, out_tsv):
    """Merge all per-cell result.json files from a sharded benchmark run.

    Run this once all SLURM shards from 'refrover benchmark --shard i/N' have
    finished. Collects every result.json under --outdir into one tidy TSV.
    """
    from refrover.mag_benchmark import aggregate_results

    out_tsv = Path(out_tsv) if out_tsv else Path(outdir) / "benchmark_results.tsv"
    df = aggregate_results(outdir)
    if df.empty:
        # Help the user distinguish "nothing ran" from "everything errored".
        import subprocess as _sp
        n_cells = len(list(Path(outdir).rglob("result.json")))
        n_err   = len(list(Path(outdir).rglob("error.json")))
        n_depth = len(list(Path(outdir).rglob("depth.txt")))
        click.echo(
            f"No output files found under {outdir}\n"
            f"  result.json files: {n_cells}\n"
            f"  error.json files:  {n_err}\n"
            f"  depth.txt files:   {n_depth}  (cells that reached MetaBAT2 input stage)\n"
            f"\nIf depth.txt > 0 but error.json = 0, the process was killed "
            f"before the error handler ran (OOM, walltime, missing tool). "
            f"Check SLURM logs or run one cell interactively:\n"
            f"  refrover benchmark --manifest samples.tsv "
            f"--jaccard-matrix work/matrices/jaccard_matrix.tsv "
            f"--assemblies-dir <asm_dir> --rules random --k-range 3 "
            f"--focals focals_1.txt --outdir work/benchmark_debug --force"
        )
        raise SystemExit(1)
    errors = df[df.get("error", pd.Series(dtype=object)).notna()] if "error" in df.columns else df.iloc[0:0]
    ok = df[~df.index.isin(errors.index)]
    click.echo(f"{len(ok)} cells succeeded, {len(errors)} failed → {out_tsv}")
    if not errors.empty:
        click.echo("\nFailed cells:")
        click.echo(errors[["matrix", "rule", "k", "focal", "error"]].to_string(index=False))
    df.to_csv(out_tsv, sep="\t", index=False)


# ── run ───────────────────────────────────────────────────────────────────────

@main.command()
@click.option("--manifest", required=True, type=click.Path(exists=True))
@click.option("--selector", default="archetype", show_default=True,
              type=click.Choice(["random", "maxmin", "kmedoids", "archetype",
                                 "greedy_var", "containment"]))
@click.option("--k", default=5, show_default=True)
@click.option("--min-jaccard", default=0.1, show_default=True)
@click.option("--containment-matrix", type=click.Path(exists=True),
              help="Cross-sample containment matrix TSV (required for --selector containment)")
@click.option("--aligner", default="bwa-mem2", show_default=True,
              type=click.Choice(["bwa-mem2", "bwa", "minimap2"]))
@click.option("--binners", default="generic", show_default=True)
@click.option("--threads", default=8, show_default=True)
@click.option("--outdir", required=True, type=click.Path())
@click.option("--force", is_flag=True)
def run(manifest, selector, k, min_jaccard, containment_matrix, aligner, binners,
        threads, outdir, force):
    """Run the full RefRover pipeline end-to-end with a single selection method."""
    from refrover.pipeline import RefRoverPipeline
    from refrover.io import read_manifest
    from refrover.selectors import SELECTOR_REGISTRY, CONTAINMENT_SELECTORS
    from refrover.similarity import load_containment_matrix

    df = read_manifest(manifest)
    sel_cls = SELECTOR_REGISTRY[selector]
    sel = sel_cls(k=k, min_jaccard=min_jaccard)
    binner_list = [b.strip() for b in binners.split(",")]

    cont_df = None
    if selector in CONTAINMENT_SELECTORS:
        if not containment_matrix:
            raise click.ClickException(
                f"--selector {selector} requires --containment-matrix"
            )
        cont_df = load_containment_matrix(containment_matrix)

    pipeline = RefRoverPipeline(
        manifest=df,
        selector=sel,
        binners=binner_list,
        threads=threads,
        outdir=Path(outdir),
        force=force,
        containment_matrix=cont_df,
    )
    results = pipeline.run()
    click.echo(f"Done. Coverage tables: {list(results.coverage_tables.keys())}")


# ── rank-selectors ────────────────────────────────────────────────────────────

# ── gtdb-gather ───────────────────────────────────────────────────────────────

@main.command(name="gtdb-gather")
@click.option("--manifest", required=True, type=click.Path(exists=True),
              help="Sample manifest TSV (sample_id, read-sketch column needed — "
                   "see --read-sketches-col)")
@click.option("--read-sketches", required=True, type=click.Path(exists=True),
              help="Directory of per-sample read .sig files (stem = sample_id); "
                   "built by 'refrover sketch-reads'")
@click.option("--gtdb-db", required=True, type=click.Path(exists=True),
              help="GTDB sourmash signature database (.zip). "
                   "Download: https://sourmash.readthedocs.io/en/latest/databases.html")
@click.option("--outdir", required=True, type=click.Path(),
              help="Directory to write per-sample {sample_id}_gather.csv files")
@click.option("--ksize", default=31, show_default=True,
              help="k-mer size; must match the read sketches and GTDB database")
@click.option("--threshold-bp", default=50_000, show_default=True,
              help="Minimum k-mer overlap (bp) to report a GTDB genome match")
@click.option("--shard", default=None,
              help="i/N — process only 1-based shard i of N samples (for SLURM arrays). "
                   "Run 'refrover gtdb-matrix' when all shards finish.")
@click.option("--force", is_flag=True, help="Re-run even if output CSV already exists")
def gtdb_gather_cmd(manifest, read_sketches, gtdb_db, outdir, ksize, threshold_bp,
                    shard, force):
    """Run sourmash gather per sample to identify GTDB taxonomic composition.

    Produces one {sample_id}_gather.csv per sample. Run 'refrover gtdb-matrix'
    once all samples are done to aggregate into the matrix used by
    'refrover benchmark --gtdb-matrix'.

    Use --shard i/N to split across SLURM array tasks.
    """
    from refrover.gtdb import run_gather_sample
    from refrover.io import read_manifest

    df = read_manifest(manifest)
    sample_ids = _shard(df["sample_id"].tolist(), shard)
    sig_dir = Path(read_sketches)
    outdir = Path(outdir)

    n_done = n_skip = n_fail = 0
    for sid in sample_ids:
        sig = sig_dir / f"{sid}.sig"
        if not sig.exists():
            click.echo(f"  WARNING: read sig not found for {sid}: {sig}", err=True)
            n_fail += 1
            continue
        result = run_gather_sample(
            sid, sig, gtdb_db, outdir,
            ksize=ksize, threshold_bp=threshold_bp, force=force,
        )
        if result is None:
            click.echo(f"  {sid}: no GTDB matches")
            n_skip += 1
        else:
            n_done += 1

    click.echo(
        f"gtdb-gather: {n_done} gathered, {n_skip} no-match, {n_fail} missing-sig "
        f"→ {outdir}"
    )


# ── gtdb-matrix ───────────────────────────────────────────────────────────────

@main.command(name="gtdb-matrix")
@click.option("--gather-dir", required=True, type=click.Path(exists=True),
              help="Directory of {sample_id}_gather.csv files (from refrover gtdb-gather)")
@click.option("--manifest", required=True, type=click.Path(exists=True),
              help="Sample manifest TSV (provides the expected sample_id list)")
@click.option("--outdir", required=True, type=click.Path(),
              help="Directory to write gtdb_matrix.tsv")
@click.option("--clade-level", default="genus", show_default=True,
              help="Taxonomic rank for clade labels: domain, phylum, class, order, "
                   "family, genus, species (or a raw GTDB prefix like 'g__')")
@click.option("--force", is_flag=True)
def gtdb_matrix_cmd(gather_dir, manifest, outdir, clade_level, force):
    """Aggregate per-sample gather CSVs into the GTDB abundance matrix.

    Reads every {sample_id}_gather.csv written by 'refrover gtdb-gather' and
    combines them into a samples × taxa TSV. Also writes clades.tsv (one column:
    the dominant genus per sample) for inspection.

    Pass the resulting gtdb_matrix.tsv to 'refrover benchmark --gtdb-matrix'.
    """
    from refrover.gtdb import aggregate_gather, infer_clades, write_gtdb_matrix
    from refrover.io import read_manifest

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    matrix_out = outdir / "gtdb_matrix.tsv"
    clades_out = outdir / "clades.tsv"

    if matrix_out.exists() and not force:
        click.echo(f"GTDB matrix already exists at {matrix_out} (use --force to redo)")
        return

    df = read_manifest(manifest)
    sample_ids = df["sample_id"].tolist()

    click.echo(f"Aggregating gather CSVs from {gather_dir} …")
    matrix = aggregate_gather(gather_dir, sample_ids=sample_ids)
    write_gtdb_matrix(matrix, matrix_out)
    click.echo(f"GTDB matrix ({matrix.shape[0]}×{matrix.shape[1]}) → {matrix_out}")

    clades = infer_clades(matrix, level=clade_level)
    clades.to_csv(clades_out, sep="\t", header=True)
    click.echo(
        f"Clade labels ({clade_level}) → {clades_out}  "
        f"({clades.nunique()} unique clades)"
    )


# ── rank-selectors ────────────────────────────────────────────────────────────

@main.command(name="rank-selectors")
@click.option("--jaccard-matrix", type=click.Path(exists=True),
              help="Precomputed Jaccard matrix TSV")
@click.option("--containment-matrix", type=click.Path(exists=True),
              help="Containment matrix TSV (alternative to --jaccard-matrix)")
@click.option("--rules", default="random,maxmin,kmedoids,greedy_var,css",
              show_default=True, help="Comma-separated rule IDs to score")
@click.option("--k-range", default="3,5,8,10", show_default=True,
              help="Comma-separated k values to test; 'adaptive' also accepted")
@click.option("--adaptive-k-method",
              default="containment_saturation", show_default=True,
              type=click.Choice(["containment_saturation", "scree_elbow",
                                 "similarity_gap", "saturation_curve"]),
              help="Estimator used when k='adaptive'")
@click.option("--min-similarity", default=0.1, show_default=True)
@click.option("--outdir", required=True, type=click.Path())
def rank_selectors_cmd(jaccard_matrix, containment_matrix, rules, k_range,
                       adaptive_k_method, min_similarity, outdir):
    """Fast alignment-free proxy scoring across selection rules.

    Ranks selection rules in seconds (no alignment) using all three proxy scores:
    frac_variance (fraction of cross-sample variance explained), effective_rank
    (number of independent axes selected), and tiered_axis_count (high- and
    medium-quality axes weighted 2:1). Use this for a cheap screen before running
    'refrover benchmark'. Provide either --jaccard-matrix or --containment-matrix.

    Writes two TSVs: selector_ranking.tsv (mean scores per rule×k) and
    selector_scores.tsv (full per-sample distribution for deeper analysis).
    """
    import pandas as pd
    from refrover.grid import FeatureSpaceSpec, headline, run_grid

    if bool(jaccard_matrix) == bool(containment_matrix):
        raise click.ClickException(
            "Provide exactly one of --jaccard-matrix or --containment-matrix."
        )

    matrix_path = jaccard_matrix or containment_matrix
    matrix_name = "jaccard" if jaccard_matrix else "containment"
    matrix = pd.read_csv(matrix_path, sep="\t", index_col=0)

    rule_list = [r.strip() for r in rules.split(",")]
    k_list = [
        k.strip() if k.strip().lower() == "adaptive" else int(k.strip())
        for k in k_range.split(",")
    ]

    spec = FeatureSpaceSpec(name=matrix_name, matrix=matrix, threshold=min_similarity)
    result = run_grid([spec], rule_list, k_list, adaptive_k_method=adaptive_k_method)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ranking = headline(result.scores)
    ranking.to_csv(outdir / "selector_ranking.tsv", sep="\t", index=False)
    result.scores.to_csv(outdir / "selector_scores.tsv", sep="\t", index=False)

    click.echo(ranking.to_string(index=False))
    click.echo(f"\nRanking → {outdir / 'selector_ranking.tsv'}")
    click.echo(f"Per-sample scores → {outdir / 'selector_scores.tsv'}")
