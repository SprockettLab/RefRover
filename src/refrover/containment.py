"""
Cross-sample containment matrix computation.

For each query (read sketch) and reference (assembly sketch), computes
containment(reads_i, assembly_j) = |hashes(reads_i) ∩ hashes(assembly_j)| /
|hashes(reads_i)| — the fraction of the query's read k-mers found in the
assembly. This is the signal ContainmentSelector selects on: it directly
predicts how well sample i's reads will map to assembly j.

The matrix is square-ish and labelled by sample_id on both axes (rows = query
read samples, columns = candidate assemblies). It need not be symmetric:
containment is directional.
"""

from pathlib import Path

import pandas as pd


def _load_minhash(sig_path: Path | str, ksize: int):
    """Load a single MinHash from a sourmash .sig file at the given ksize."""
    import sourmash

    sig = next(sourmash.load_file_as_signatures(str(sig_path), ksize=ksize))
    return sig.minhash


# Intermediate suffixes added by common assemblers before the file extension.
# e.g. megahit: "sample.contigs.fasta" -> strip ".contigs" -> sample_id "sample"
_ASSEMBLER_SUFFIXES = (".contigs", ".scaffolds", ".assembly", ".final", ".fa", ".fasta", ".fna")


def _sig_stem(p: Path) -> str:
    """
    Derive sample_id from a .sig filename by stripping the .sig extension and
    any assembler-specific intermediate suffix (e.g. '.contigs' from megahit).
    """
    stem = p.stem  # removes ".sig"
    for suffix in _ASSEMBLER_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem


def sigs_by_sample(sketch_dir: Path | str) -> dict[str, Path]:
    """Map sample_id -> .sig path for every sketch in a directory.

    The sample_id is the filename stem after stripping the .sig extension and
    any assembler-specific intermediate suffix (e.g. megahit's '.contigs').
    """
    sketch_dir = Path(sketch_dir)
    return {_sig_stem(p): p for p in sorted(sketch_dir.glob("*.sig"))}


def compute_containment_matrix(
    read_sigs: dict[str, Path | str],
    assembly_sigs: dict[str, Path | str],
    *,
    ksize: int = 31,
) -> pd.DataFrame:
    """
    Compute the reads-vs-assemblies containment matrix.

    Parameters
    ----------
    read_sigs : dict
        sample_id -> read sketch (.sig) path.
    assembly_sigs : dict
        sample_id -> assembly sketch (.sig) path.
    ksize : int
        k-mer size; read and assembly sketches must share it.

    Returns
    -------
    pd.DataFrame
        Rows = read sample_ids (sorted), columns = assembly sample_ids (sorted).
        Values = containment(reads_row, assembly_col) in [0, 1]. Index name is
        'query_id', matching load_containment_matrix.
    """
    asm_ids = sorted(assembly_sigs)
    asm_mh = {a: _load_minhash(assembly_sigs[a], ksize) for a in asm_ids}

    rows: dict[str, dict[str, float]] = {}
    for q in sorted(read_sigs):
        read_mh = _load_minhash(read_sigs[q], ksize)
        rows[q] = {a: float(read_mh.contained_by(asm_mh[a])) for a in asm_ids}

    df = pd.DataFrame.from_dict(rows, orient="index")
    df = df.reindex(columns=asm_ids)
    df.index.name = "query_id"
    return df


def write_containment_matrix(matrix: pd.DataFrame, path: Path | str) -> Path:
    """Write a containment matrix to TSV (the format load_containment_matrix reads)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(path, sep="\t")
    return path
