"""
Load and operate on pairwise Jaccard similarity matrices.

The primary input format is the CSV written by `sourmash compare --csv`.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def load_similarity_matrix(csv_path: Path | str) -> pd.DataFrame:
    """
    Load a sourmash compare CSV into a labelled similarity DataFrame.

    sourmash compare --csv writes:
      - Row 0: comma-separated sample labels (from sig names or filenames)
      - Rows 1..n: the matrix rows, one per sample

    Returns a square DataFrame with sample IDs as both index and columns.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, index_col=0)
    # The first column after the index is also a label column in sourmash output;
    # ensure the matrix is square and symmetric.
    if df.shape[0] != df.shape[1]:
        raise ValueError(
            f"Similarity matrix is not square: {df.shape}. "
            "Expected output from `sourmash compare --csv`."
        )
    df.index.name = None
    df.columns.name = None
    return df


def jaccard_to_distance(sim_matrix: pd.DataFrame) -> pd.DataFrame:
    """Convert a Jaccard similarity matrix to a distance matrix (1 - Jaccard)."""
    return 1.0 - sim_matrix


def load_containment_matrix(tsv_path: Path | str) -> pd.DataFrame:
    """
    Load a cross-sample containment matrix (TSV) for ContainmentSelector.

    Expected layout (as written by compute_containment.py):
      - First column: query sample_id (the row index).
      - Remaining columns: one per candidate assembly, labelled by sample_id.
      - Values: containment(reads_row, assembly_col) in [0, 1].

    The matrix is square and labelled by sample_id on both axes, but need not be
    symmetric (containment is directional). Returns a labelled DataFrame.
    """
    tsv_path = Path(tsv_path)
    df = pd.read_csv(tsv_path, sep="\t", index_col=0)
    df.index.name = None
    df.columns.name = None
    return df


def filter_by_jaccard(
    sim_matrix: pd.DataFrame,
    query_id: str,
    min_jaccard: float,
) -> list[str]:
    """
    Return sample IDs with Jaccard(query, candidate) >= min_jaccard.

    The query itself is always included (diagonal = 1.0).
    Raises KeyError if query_id is not in the matrix.
    """
    if query_id not in sim_matrix.index:
        raise KeyError(f"query_id '{query_id}' not found in similarity matrix")
    row = sim_matrix.loc[query_id]
    return row[row >= min_jaccard].index.tolist()


def matrix_from_sigs(
    sig_paths: "list[Path]",
    *,
    ksize: int = 31,
) -> pd.DataFrame:
    """
    Compute a pairwise Jaccard similarity matrix directly from sourmash .sig files,
    without calling the sourmash CLI.

    Uses the sourmash Python API for speed (no subprocess overhead).
    Loads all sketches into memory then computes all n*(n-1)/2 pairs.

    Returns a labelled DataFrame (index and columns = sig names).
    """
    import sourmash

    sigs = []
    for p in sig_paths:
        sig = next(sourmash.load_file_as_signatures(str(p), ksize=ksize))
        sigs.append(sig)

    n = len(sigs)
    names = [s.name or Path(sig_paths[i]).stem for i, s in enumerate(sigs)]
    mat = np.eye(n, dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            j_val = sigs[i].minhash.jaccard(sigs[j].minhash)
            mat[i, j] = j_val
            mat[j, i] = j_val

    return pd.DataFrame(mat, index=names, columns=names)
