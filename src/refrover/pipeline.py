"""
RefRoverPipeline: orchestrates the full sketch → select → align → coverage → format pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd

from refrover.selectors.base import BaseSelector


@dataclass
class PipelineResults:
    coverage_tables: dict[str, Path] = field(default_factory=dict)
    assignments: pd.DataFrame = field(default_factory=pd.DataFrame)


class RefRoverPipeline:
    def __init__(
        self,
        manifest: pd.DataFrame,
        selector: BaseSelector,
        binners: list[str],
        threads: int = 8,
        outdir: Path = Path("refrover_out"),
        force: bool = False,
    ):
        self.manifest = manifest
        self.selector = selector
        self.binners = binners
        self.threads = threads
        self.outdir = Path(outdir)
        self.force = force

    def run(self) -> PipelineResults:
        from refrover.sketch import sketch_assemblies, compare_sketches
        from refrover.similarity import load_similarity_matrix
        from refrover.io import write_assignments
        from refrover.align import run_alignment
        from refrover.coverage import run_coverm, load_coverage
        from refrover.formatters import format_for_binner

        outdir = self.outdir
        sketch_dir = outdir / "sketches"
        bam_dir = outdir / "bams"
        cov_dir = outdir / "coverage"
        fmt_dir = outdir / "formatted"

        # 1. Sketch
        sig_paths = sketch_assemblies(
            self.manifest["assembly"].tolist(),
            outdir=sketch_dir,
            threads=self.threads,
            force=self.force,
        )

        # 2. Pairwise similarity
        sim_csv = outdir / "similarity.csv"
        compare_sketches(sig_paths, output_csv=sim_csv, force=self.force)
        sim_matrix = load_similarity_matrix(sim_csv)

        # 3. Prototype selection
        rows = []
        for _, row in self.manifest.iterrows():
            sid = row["sample_id"]
            prototypes = self.selector.select(sim_matrix, sid)
            rows.append({"sample_id": sid, "prototype_ids": ",".join(prototypes),
                          "n_prototypes": len(prototypes)})
        assignments = pd.DataFrame(rows)
        write_assignments(assignments, outdir / "assignments.tsv")

        # 4. Align
        run_alignment(
            self.manifest, assignments,
            outdir=bam_dir, threads=self.threads, force=self.force,
        )

        # 5. Coverage
        cov_tsv = run_coverm(bam_dir, outdir=cov_dir, threads=self.threads, force=self.force)
        coverage_df = load_coverage(cov_tsv)

        # 6. Format
        fmt_dir.mkdir(parents=True, exist_ok=True)
        coverage_tables = {}
        for binner in self.binners + ["generic"]:
            out = format_for_binner(coverage_df, binner=binner, outdir=fmt_dir, force=self.force)
            coverage_tables[binner] = out if isinstance(out, Path) else out[0]

        return PipelineResults(coverage_tables=coverage_tables, assignments=assignments)
