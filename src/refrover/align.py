"""
Alignment wrappers: BWA-MEM2, BWA, minimap2.

For each sample, concatenates its assigned prototype assemblies into a
combined reference, builds an index, aligns reads, and writes a sorted BAM.
"""

import logging
import subprocess
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def run_alignment(
    manifest: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    outdir: Path | str,
    aligner: str = "bwa-mem2",
    threads: int = 8,
    force: bool = False,
    bwa_path: str = "bwa-mem2",
    minimap2_path: str = "minimap2",
    samtools_path: str = "samtools",
) -> list[Path]:
    """
    For each sample, align reads against its assigned prototype assemblies.

    Steps per sample:
      1. Concatenate prototype FASTAs into a combined reference.
      2. Build aligner index (bwa-mem2 index or minimap2 index).
      3. Align reads (paired/single/long).
      4. Sort and index the BAM.

    Returns list of sorted BAM paths, in manifest order.
    Skips existing BAMs unless force=True.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    asgn_map = dict(zip(assignments["sample_id"], assignments["prototype_ids"]))
    bam_paths: list[Path] = []

    for _, row in manifest.iterrows():
        sid = row["sample_id"]
        if sid not in asgn_map:
            log.warning("No assignment for sample %s — skipping", sid)
            continue

        bam_path = outdir / f"{sid}.sorted.bam"
        bam_paths.append(bam_path)

        if bam_path.exists() and not force:
            log.debug("Skipping %s (BAM exists)", sid)
            continue

        prototype_ids = [p.strip() for p in asgn_map[sid].split(",")]
        sample_dir = outdir / sid
        sample_dir.mkdir(exist_ok=True)

        # Locate prototype FASTA paths from manifest
        proto_fastas = _resolve_fastas(manifest, prototype_ids, row["assembly"])

        # Build combined reference
        ref_fa = sample_dir / "reference.fasta"
        _concat_fastas(proto_fastas, ref_fa)

        # Align
        if aligner in ("bwa-mem2", "bwa"):
            _align_bwa(
                row, ref_fa, bam_path, sample_dir,
                aligner_bin=bwa_path if aligner == "bwa-mem2" else "bwa",
                samtools_bin=samtools_path,
                threads=threads,
            )
        elif aligner == "minimap2":
            _align_minimap2(
                row, ref_fa, bam_path, sample_dir,
                minimap2_bin=minimap2_path,
                samtools_bin=samtools_path,
                threads=threads,
            )
        else:
            raise ValueError(f"Unknown aligner: {aligner!r}")

        log.info("Aligned %s → %s", sid, bam_path)

    return bam_paths


# ── Internal helpers ──────────────────────────────────────────────────────────

def _resolve_fastas(
    manifest: pd.DataFrame,
    prototype_ids: list[str],
    own_assembly: str,
) -> list[Path]:
    """Return FASTA paths for the given prototype IDs, using the manifest for lookup."""
    asm_map = dict(zip(manifest["sample_id"], manifest["assembly"]))
    paths = []
    for pid in prototype_ids:
        if pid in asm_map:
            paths.append(Path(asm_map[pid]))
        else:
            # Fallback: assume the prototype ID is already a path
            paths.append(Path(pid))
    return paths


def _concat_fastas(fastas: list[Path], out_fa: Path) -> None:
    """Concatenate FASTA files into a single reference."""
    with open(out_fa, "wb") as fout:
        for fa in fastas:
            with open(fa, "rb") as f:
                fout.write(f.read())


def _run(cmd: list[str], description: str) -> None:
    """Run a subprocess command; raise RuntimeError on failure."""
    log.debug("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"{description} failed (exit {result.returncode}):\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  stderr: {result.stderr[:500]}"
        )


def _align_bwa(
    sample_row: pd.Series,
    ref_fa: Path,
    out_bam: Path,
    work_dir: Path,
    aligner_bin: str,
    samtools_bin: str,
    threads: int,
) -> None:
    """Index with bwa-mem2/bwa, align, sort into out_bam."""
    # Index
    _run([aligner_bin, "index", str(ref_fa)], f"bwa index {ref_fa.name}")

    # Build align command
    r1 = sample_row.get("r1")
    r2 = sample_row.get("r2") if "r2" in sample_row.index else None
    long_reads = sample_row.get("long_reads") if "long_reads" in sample_row.index else None

    sam_path = work_dir / "aln.sam"
    rg = f"@RG\\tID:{sample_row['sample_id']}\\tSM:{sample_row['sample_id']}"

    if pd.notna(r1):
        mem_cmd = [aligner_bin, "mem", "-t", str(threads), "-R", rg, str(ref_fa), str(r1)]
        if pd.notna(r2):
            mem_cmd.append(str(r2))
        with open(sam_path, "w") as fout:
            result = subprocess.run(mem_cmd, stdout=fout, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"bwa mem failed:\n{result.stderr[:500]}")

        if pd.notna(long_reads):
            _merge_long_reads_bwa(sample_row, ref_fa, sam_path, work_dir, threads)

    elif pd.notna(long_reads):
        # Long-reads only: fall through to minimap2
        raise ValueError(
            f"Sample {sample_row['sample_id']}: long-reads-only with bwa aligner. "
            "Use --aligner minimap2 for long-read or hybrid samples."
        )

    _sort_index(sam_path, out_bam, samtools_bin, threads)


def _merge_long_reads_bwa(
    sample_row: pd.Series,
    ref_fa: Path,
    short_sam: Path,
    work_dir: Path,
    threads: int,
    minimap2_bin: str = "minimap2",
) -> None:
    """
    For hybrid samples under a bwa aligner: align the long reads with minimap2
    (bwa cannot map long reads) and append them to the short-read SAM, producing
    one merged alignment per sample. Modifies short_sam in place.
    """
    long_sam = work_dir / "long.sam"
    # _align_minimap2_preset redirects minimap2's SAM to long_sam; _run does not
    # (it captures stdout), which is why the long alignments must go through the
    # preset helper here.
    _align_minimap2_preset(
        minimap2_bin, ref_fa, [str(sample_row["long_reads"])], long_sam,
        preset="map-ont", threads=threads,
    )
    # Append long-read alignments, skipping the long SAM's header (the short
    # SAM already has a header for the same reference).
    with open(short_sam, "a") as fout, open(long_sam) as fin:
        for line in fin:
            if not line.startswith("@"):
                fout.write(line)


def _align_minimap2(
    sample_row: pd.Series,
    ref_fa: Path,
    out_bam: Path,
    work_dir: Path,
    minimap2_bin: str,
    samtools_bin: str,
    threads: int,
) -> None:
    """Align with minimap2 (long reads or hybrid), sort into out_bam."""
    r1 = sample_row.get("r1") if "r1" in sample_row.index else None
    long_reads = sample_row.get("long_reads") if "long_reads" in sample_row.index else None
    sam_path = work_dir / "aln.sam"

    if pd.notna(long_reads):
        preset = "map-ont"  # or map-pb for PacBio
        reads = [str(long_reads)]
        if pd.notna(r1):
            # Hybrid: align short reads with sr preset, long with map-ont, then merge
            _align_minimap2_preset(minimap2_bin, ref_fa, [str(r1)], work_dir / "short.sam",
                                   preset="sr", threads=threads)
            _align_minimap2_preset(minimap2_bin, ref_fa, reads, work_dir / "long.sam",
                                   preset=preset, threads=threads)
            _merge_sams([work_dir / "short.sam", work_dir / "long.sam"], sam_path)
        else:
            _align_minimap2_preset(minimap2_bin, ref_fa, reads, sam_path,
                                   preset=preset, threads=threads)
    elif pd.notna(r1):
        reads = [str(r1)]
        r2 = sample_row.get("r2") if "r2" in sample_row.index else None
        if pd.notna(r2):
            reads.append(str(r2))
        _align_minimap2_preset(minimap2_bin, ref_fa, reads, sam_path,
                               preset="sr", threads=threads)
    else:
        raise ValueError(f"Sample {sample_row['sample_id']}: no reads to align")

    _sort_index(sam_path, out_bam, samtools_bin, threads)


def _align_minimap2_preset(
    minimap2_bin: str,
    ref_fa: Path,
    reads: list[str],
    out_sam: Path,
    preset: str,
    threads: int,
) -> None:
    cmd = [minimap2_bin, "-ax", preset, "-t", str(threads), str(ref_fa)] + reads
    with open(out_sam, "w") as fout:
        result = subprocess.run(cmd, stdout=fout, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"minimap2 ({preset}) failed:\n{result.stderr[:500]}")


def _merge_sams(sam_paths: list[Path], out_sam: Path) -> None:
    """Merge SAM files, keeping header only from the first."""
    with open(out_sam, "w") as fout:
        for i, p in enumerate(sam_paths):
            with open(p) as fin:
                for line in fin:
                    if i == 0 or not line.startswith("@"):
                        fout.write(line)


def _sort_index(sam_path: Path, out_bam: Path, samtools_bin: str, threads: int) -> None:
    """Sort SAM → BAM and index."""
    _run(
        [samtools_bin, "sort", "-@", str(threads), "-o", str(out_bam), str(sam_path)],
        f"samtools sort → {out_bam.name}",
    )
    _run([samtools_bin, "index", str(out_bam)], f"samtools index {out_bam.name}")
