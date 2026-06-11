import numpy as np
import pandas as pd
import pytest
from pathlib import Path


def _make_sim(matrix: np.ndarray, ids: list[str]) -> pd.DataFrame:
    np.fill_diagonal(matrix, 1.0)
    sym = (matrix + matrix.T) / 2
    np.fill_diagonal(sym, 1.0)
    return pd.DataFrame(sym, index=ids, columns=ids)


@pytest.fixture
def clustered_sim():
    """
    12 samples in 3 clusters of 4.
    Within-cluster Jaccard ~ 0.6-0.8, cross-cluster ~ 0.05.
    """
    rng = np.random.default_rng(42)
    n = 12
    ids = [f"s{i:02d}" for i in range(n)]
    m = np.full((n, n), 0.05)
    for start in (0, 4, 8):
        for i in range(start, start + 4):
            for j in range(start, start + 4):
                if i != j:
                    m[i, j] = 0.6 + 0.2 * rng.random()
    return _make_sim(m, ids)


@pytest.fixture
def uniform_sim():
    """10 samples, all pairs roughly equidistant (Jaccard 0.1-0.4)."""
    rng = np.random.default_rng(0)
    n = 10
    ids = [f"s{i:02d}" for i in range(n)]
    m = rng.uniform(0.1, 0.4, (n, n))
    return _make_sim(m, ids)


@pytest.fixture
def sparse_sim():
    """8 samples, most pairs below min_jaccard=0.1 (stress-tests fallback)."""
    rng = np.random.default_rng(7)
    n = 8
    ids = [f"s{i:02d}" for i in range(n)]
    m = rng.uniform(0.01, 0.09, (n, n))
    # s00 has a few valid neighbours
    m[0, 1] = m[1, 0] = 0.3
    m[0, 2] = m[2, 0] = 0.2
    return _make_sim(m, ids)


@pytest.fixture
def coverage_df():
    """
    Minimal coverage DataFrame in RefRover's normalized schema: 5 contigs x
    3 samples, with a 'length' column, a '{sample}_depth' mean column, and a
    paired '{sample}_var' variance column per sample.
    """
    contigs = [f"contig_{i}" for i in range(5)]
    data = {
        "length": [1000, 2000, 500, 3000, 1500],
        "s1_depth": [10.2, 0.0, 5.5, 22.1, 8.8],
        "s2_depth": [0.0, 15.3, 6.1, 18.9, 0.0],
        "s3_depth": [7.7, 12.0, 0.0, 25.4, 3.3],
        "s1_var": [2.1, 0.0, 1.2, 4.4, 1.8],
        "s2_var": [0.0, 3.3, 1.5, 3.9, 0.0],
        "s3_var": [1.7, 2.4, 0.0, 5.1, 0.6],
    }
    return pd.DataFrame(data, index=contigs)


@pytest.fixture
def tiny_manifest(tmp_path):
    """Write a minimal valid manifest TSV and return its path."""
    content = "sample_id\tassembly\tr1\n"
    content += "S1\tassemblies/S1.fasta\treads/S1_R1.fastq.gz\n"
    content += "S2\tassemblies/S2.fasta\treads/S2_R1.fastq.gz\n"
    p = tmp_path / "manifest.tsv"
    p.write_text(content)
    return p
