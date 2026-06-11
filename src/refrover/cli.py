"""
RefRover CLI entry point.

Subcommands:
  sketch    Sketch assemblies with sourmash
  select    Prototype selection (assigns prototypes per sample)
  align     Align reads to selected prototypes
  coverage  Compute per-contig depth with CoverM
  format    Format coverage tables for downstream binners
  run       Full pipeline end-to-end
  benchmark Compare selectors on a dataset with known ground truth
"""

import click
from pathlib import Path


@click.group()
@click.version_option()
def main():
    """RefRover: differential coverage tables for metagenomic binners."""


# ── sketch ────────────────────────────────────────────────────────────────────

@main.command()
@click.option("--manifest", required=True, type=click.Path(exists=True), help="Sample manifest TSV")
@click.option("--outdir", required=True, type=click.Path(), help="Output directory for .sig files")
@click.option("--ksize", default=31, show_default=True)
@click.option("--scaled", default=1000, show_default=True)
@click.option("--threads", default=1, show_default=True)
@click.option("--force", is_flag=True, help="Re-sketch even if output exists")
def sketch(manifest, outdir, ksize, scaled, threads, force):
    """Sketch assemblies with sourmash sketch dna."""
    from refrover.io import read_manifest
    from refrover.sketch import sketch_assemblies

    df = read_manifest(manifest)
    sig_paths = sketch_assemblies(
        df["assembly"].tolist(),
        outdir=outdir,
        ksize=ksize,
        scaled=scaled,
        threads=threads,
        force=force,
    )
    click.echo(f"Sketched {len(sig_paths)} assemblies → {outdir}")


# ── select ────────────────────────────────────────────────────────────────────

@main.command()
@click.option("--sketches", type=click.Path(exists=True),
              help="Directory of assembly .sig files (Jaccard selectors)")
@click.option("--containment-matrix", type=click.Path(exists=True),
              help="Cross-sample containment matrix TSV (required for --selector containment)")
@click.option("--manifest", required=True, type=click.Path(exists=True))
@click.option("--selector", default="archetype", show_default=True,
              type=click.Choice(["random", "maxmin", "kmedoids", "archetype",
                                 "greedy_var", "containment"]))
@click.option("--k", default="5", show_default=True,
              help="Prototypes per sample, or 'auto' to infer from similarity structure")
@click.option("--adaptive-k-method", default="similarity_gap", show_default=True,
              type=click.Choice(["scree_elbow", "similarity_gap", "saturation_curve"]))
@click.option("--min-jaccard", default=0.1, show_default=True,
              help="Minimum similarity/containment floor for candidate assemblies")
@click.option("--outdir", required=True, type=click.Path())
@click.option("--force", is_flag=True)
def select(sketches, containment_matrix, manifest, selector, k, adaptive_k_method,
           min_jaccard, outdir, force):
    """Select prototype assemblies for each sample."""
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

    # Containment selectors run on a reads-vs-assembly containment matrix;
    # Jaccard selectors run on an assembly-vs-assembly matrix from the sketches.
    if selector in CONTAINMENT_SELECTORS:
        if not containment_matrix:
            raise click.ClickException(
                f"--selector {selector} requires --containment-matrix"
            )
        click.echo(f"Loading containment matrix from {containment_matrix}...")
        sim_matrix = load_containment_matrix(containment_matrix)
    else:
        if not sketches:
            raise click.ClickException(
                f"--selector {selector} requires --sketches"
            )
        sig_dir = Path(sketches)
        sig_paths = sorted(sig_dir.glob("*.sig"))
        if not sig_paths:
            raise click.ClickException(f"No .sig files found in {sketches}")
        click.echo(f"Computing similarity matrix from {len(sig_paths)} sketches...")
        sim_matrix = matrix_from_sigs(sig_paths)

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
    """Align reads to selected prototype assemblies."""
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


# ── run ───────────────────────────────────────────────────────────────────────

@main.command()
@click.option("--manifest", required=True, type=click.Path(exists=True))
@click.option("--selector", default="archetype", show_default=True,
              type=click.Choice(["random", "maxmin", "kmedoids", "archetype",
                                 "greedy_var", "containment"]))
@click.option("--k", default=5, show_default=True)
@click.option("--min-jaccard", default=0.1, show_default=True,
              help="Minimum similarity/containment floor for candidate assemblies")
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
    """Run the full RefRover pipeline end-to-end."""
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


# ── benchmark ─────────────────────────────────────────────────────────────────

@main.command()
@click.option("--manifest", required=True, type=click.Path(exists=True))
@click.option("--truth", required=True, type=click.Path(exists=True),
              help="Ground truth community TSV")
@click.option("--selectors", default="random,maxmin,kmedoids,archetype",
              show_default=True, help="Comma-separated selector IDs to benchmark")
@click.option("--k-range", default="3,5,8,10", show_default=True,
              help="Comma-separated k values to test")
@click.option("--checkm2-db", type=click.Path(exists=True), default=None,
              help="Path to CheckM2 database (required for MAG quality evaluation)")
@click.option("--outdir", required=True, type=click.Path())
@click.option("--threads", default=8, show_default=True)
def benchmark(manifest, truth, selectors, k_range, checkm2_db, outdir, threads):
    """Benchmark selectors against each other on a labelled dataset."""
    from refrover.benchmark import run_benchmark
    from refrover.io import read_manifest

    df = read_manifest(manifest)
    selector_list = [s.strip() for s in selectors.split(",")]
    k_list = [int(k.strip()) for k in k_range.split(",")]

    run_benchmark(
        manifest=df,
        truth_path=truth,
        selectors=selector_list,
        k_values=k_list,
        checkm2_db=checkm2_db,
        outdir=Path(outdir),
        threads=threads,
    )
    click.echo(f"Benchmark results → {outdir}")
