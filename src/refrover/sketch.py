"""
Wrappers around sourmash for sketching assemblies and computing pairwise similarity.
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
    Sketch each FASTA with sourmash sketch dna.

    Returns a list of .sig paths (one per input), in the same order.
    Skips existing .sig files unless force=True.
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

    # sourmash sketch dna processes multiple files in one call
    fasta_list = [str(fa) for fa, _ in to_sketch]

    cmd = [
        sourmash_path, "sketch", "dna",
        f"--param-string=k={ksize},scaled={scaled}",
        "--output-dir", str(outdir),
        "-p", f"k={ksize},scaled={scaled}",
    ] + fasta_list

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"sourmash sketch failed:\n{result.stderr}"
        )

    return all_sigs


def compare_sketches(
    sig_paths: Sequence[Path | str],
    output_csv: Path | str,
    *,
    ksize: int = 31,
    force: bool = False,
    sourmash_path: str = "sourmash",
) -> Path:
    """
    Run sourmash compare on a list of .sig files to get a pairwise CSV matrix.

    Returns the path to the output CSV.
    Raises RuntimeError on failure.
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
        raise RuntimeError(
            f"sourmash compare failed:\n{result.stderr}"
        )

    return output_csv
