import numpy as np
import pandas as pd
import pytest


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
