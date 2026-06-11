"""
Benchmark selector strategies against each other.

Two tiers:

Tier 1 — variance-explained proxy (cheap, alignment-free).
    For each (selector, k), run the selector for every sample and measure the
    fraction of total cross-sample variance the selected prototypes explain,
    via orthogonal projection of the full matrix onto the selected columns
    (so correlated/redundant selections are not double-counted). This ranks
    selectors in seconds from a similarity or containment matrix alone, with no
    alignment. See rank_selectors().

Tier 2 — MAG yield per CPU-hour (gold standard, expensive).
    Run the full pipeline + a binner + CheckM2 and count passing MAGs per
    compute. Not yet implemented (needs CheckM2 and a labelled community);
    run_benchmark() is the entry point.

The research question is whether the Tier-1 ranking predicts the Tier-2 ranking.
If it does, selector choice can be made cheaply from Tier 1 alone.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from refrover.selectors import SELECTOR_REGISTRY

# Selectors that can't run unattended in a benchmark (unimplemented).
_SKIP_SELECTORS = {"feedback"}
DEFAULT_SELECTORS = ["random", "maxmin", "kmedoids", "archetype", "greedy_var", "containment"]


def unique_variance_explained(matrix: pd.DataFrame, selected_ids: list[str]) -> float:
    """
    Fraction of total cross-sample column variance explained by selected columns.

    The full matrix (samples × features) is mean-centred; the selected columns
    span a subspace; we project all columns onto it and report
    (total_var − residual_var) / total_var. Orthogonal projection means columns
    correlated with the selected set count once, not multiple times — so this
    rewards prototypes that span independent axes of variation rather than
    piling onto the same one.

    Returns a value in [0, 1] (1.0 when the selected columns span the column
    space, e.g. all columns selected).
    """
    cols = list(matrix.columns)
    pos = {c: i for i, c in enumerate(cols)}
    idx = [pos[s] for s in selected_ids if s in pos]

    X = matrix.to_numpy(dtype=float)
    X = X - X.mean(axis=0)
    total_var = float(np.var(X, axis=0, ddof=1).sum())
    if total_var == 0.0 or not idx:
        return 0.0

    Xp = X[:, idx]
    Q, _ = np.linalg.qr(Xp)  # orthonormal basis for the selected subspace
    resid = X - Q @ (Q.T @ X)
    resid_var = float(np.var(resid, axis=0, ddof=1).sum())
    return (total_var - resid_var) / total_var


def rank_selectors(
    matrix: pd.DataFrame,
    *,
    selectors: list[str] | None = None,
    k_values: list[int],
    min_similarity: float = 0.1,
    query_ids: list[str] | None = None,
) -> pd.DataFrame:
    """
    Rank selectors by the Tier-1 variance-explained proxy on a single matrix.

    The same matrix (Jaccard similarity or containment) is fed to every selector,
    so the comparison is apples-to-apples. Selectors that can't run in the
    environment (e.g. archetype without the `archetypes` package) are skipped
    with a warning rather than aborting the whole run.

    Returns a DataFrame: selector, k, mean_var_explained, median_var_explained,
    n_samples — sorted by k then descending mean_var_explained.
    """
    selectors = selectors or DEFAULT_SELECTORS
    query_ids = query_ids if query_ids is not None else list(matrix.index)

    records = []
    for name in selectors:
        if name in _SKIP_SELECTORS or name not in SELECTOR_REGISTRY:
            warnings.warn(f"Skipping selector '{name}' (unavailable).", stacklevel=2)
            continue
        cls = SELECTOR_REGISTRY[name]
        selector_failed = False

        for k in k_values:
            sel = cls(k=k, min_jaccard=min_similarity)
            # Make the random baseline reproducible across the benchmark.
            if hasattr(sel, "seed") and getattr(sel, "seed") is None:
                sel.seed = 0

            scores = []
            for q in query_ids:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # fewer-than-k candidate notices
                    try:
                        chosen = sel.select(matrix, q)
                    except (ImportError, NotImplementedError) as exc:
                        warnings.warn(
                            f"Skipping selector '{name}': {exc}", stacklevel=2
                        )
                        selector_failed = True
                        break
                    except (KeyError, ValueError):
                        continue
                scores.append(unique_variance_explained(matrix, chosen))

            if selector_failed:
                break
            if scores:
                records.append({
                    "selector": name,
                    "k": k,
                    "mean_var_explained": float(np.mean(scores)),
                    "median_var_explained": float(np.median(scores)),
                    "n_samples": len(scores),
                })

    df = pd.DataFrame.from_records(
        records,
        columns=["selector", "k", "mean_var_explained", "median_var_explained", "n_samples"],
    )
    if not df.empty:
        df = df.sort_values(["k", "mean_var_explained"], ascending=[True, False])
    return df.reset_index(drop=True)


def run_benchmark(
    manifest: pd.DataFrame,
    truth_path: Path | str,
    selectors: list[str],
    k_values: list[int],
    outdir: Path | str,
    threads: int = 8,
    checkm2_db: Path | str | None = None,
) -> pd.DataFrame:
    """
    Tier 2 — MAG yield per CPU-hour against a labelled ground-truth community.

    For each (selector, k): run RefRoverPipeline, bin with MetaBAT2, evaluate with
    CheckM2, and report passing MAGs per compute. Not yet implemented.
    """
    raise NotImplementedError(
        "Tier-2 benchmark (CheckM2 MAG yield) not yet implemented. "
        "Use rank_selectors() for the alignment-free Tier-1 variance proxy."
    )
