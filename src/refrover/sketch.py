"""
Wrappers around sourmash for sketching assemblies and reads.
"""

import subprocess
from pathlib import Path
from typing import Sequence


def sketch_assemblies(
    fasta_paths: Sequence[Path | str],
    outdir: Path | str,
    *,
    ksize: int = 31,
    scaled: int = 1000,
    threads: int = 1,
    force: bool = False,
    sourmash_path: str = "sourmash",
) -> list[Path]:
    """
    Sketch each assembly FASTA with sourmash sketch dna.

    All FASTAs are processed in a single subprocess call (sourmash handles
    batching internally). Skips already-existing .sig files unless force=True.
    Returns one .sig path per input, in the same order.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    to_sketch: list[tuple[Path, Path]] = []
    all_sigs: list[Path] = []

    for fa in fasta_paths:
        fa = Path(fa)
        sig_path = outdir / (fa.stem + ".sig")
        all_sigs.append(sig_path)
        if sig_path.exists() and not force:
            continue
        to_sketch.append((fa, sig_path))

    if not to_sketch:
        return all_sigs

    cmd = [
        sourmash_path, "sketch", "dna",
        "-p", f"k={ksize},scaled={scaled}",
        "--output-dir", str(outdir),
    ] + [str(fa) for fa, _ in to_sketch]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"sourmash sketch (assemblies) failed:\n{result.stderr}")

    return all_sigs


def sketch_reads(
    sample_id: str,
    read_files: Sequence[Path | str],
    outdir: Path | str,
    *,
    ksize: int = 31,
    scaled: int = 1000,
    force: bool = False,
    sourmash_path: str = "sourmash",
) -> Path:
    """
    Sketch all reads from one sample into a single merged .sig.

    Multiple read files (r1, r2, long_reads) are merged under ``sample_id`` so
    containment(reads_i, assembly_j) is computed over all of a sample's reads.
    Skips if the .sig already exists unless force=True.

    Parameters
    ----------
    sample_id : str
        Identifier used as both the sketch name and the output filename stem.
    read_files : sequence of paths
        FASTQ (or FASTQ.gz) files for this sample; None/empty-string entries
        are silently skipped (handles optional r2 / long_reads columns).
    outdir : path
        Directory to write ``{sample_id}.sig`` into.

    Returns
    -------
    Path
        Path to the output .sig file.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sig_path = outdir / f"{sample_id}.sig"

    if sig_path.exists() and not force:
        return sig_path

    inputs = [str(r) for r in read_files if r]
    if not inputs:
        raise ValueError(f"No read files provided for sample '{sample_id}'")

    cmd = [
        sourmash_path, "sketch", "dna",
        "-p", f"k={ksize},scaled={scaled}",
        "--merge", sample_id,
        "--output", str(sig_path),
    ] + inputs

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"sourmash sketch (reads) failed for '{sample_id}':\n{result.stderr}"
        )
    return sig_path


def compare_sketches(
    sig_paths: Sequence[Path | str],
    output_csv: Path | str,
    *,
    ksize: int = 31,
    force: bool = False,
    sourmash_path: str = "sourmash",
) -> Path:
    """
    Run sourmash compare on a list of .sig files → pairwise Jaccard CSV.

    For a labeled DataFrame (sample_id on both axes) use
    ``refrover jaccard`` (CLI) or ``feature_spaces.jaccard_matrix()`` instead;
    this function is a thin sourmash compare wrapper kept for scripting.
    """
    output_csv = Path(output_csv)
    if output_csv.exists() and not force:
        return output_csv

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sourmash_path, "compare",
        "--csv", str(output_csv),
        f"--ksize={ksize}",
    ] + [str(p) for p in sig_paths]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"sourmash compare failed:\n{result.stderr}")

    return output_csv
