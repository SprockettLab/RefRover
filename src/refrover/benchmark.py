"""
Alignment-free proxy scoring for selection methods.

For each (rule, k), runs the rule for every sample against the provided
similarity/containment matrix and computes how much cross-sample variance the
selected references explain, via orthogonal projection (correlated/redundant
picks don't double-count). This ranks rules in seconds with no alignment.

Use ``rank_selectors()`` here when you want a fast, cheap screen across rules
before committing to the real benchmark (``mag_benchmark.run_pilot``). The
research question is whether this proxy ranking predicts the real MAG-yield
ranking from ``mag_benchmark``.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from refrover.selectors import SELECTOR_REGISTRY

# Canonical implementation lives in refrover.scores; re-exported here under the
# historical name for existing callers.
from refrover.scores import frac_variance as unique_variance_explained  # noqa: F401

_SKIP_SELECTORS = {"feedback"}
DEFAULT_SELECTORS = ["random", "maxmin", "kmedoids", "archetype", "greedy_var", "containment"]


def rank_selectors(
    matrix: pd.DataFrame,
    *,
    selectors: list[str] | None = None,
    k_values: list[int],
    min_similarity: float = 0.1,
    query_ids: list[str] | None = None,
) -> pd.DataFrame:
    """
    Rank selection rules by the alignment-free variance-explained proxy.

    The same matrix (Jaccard similarity or containment) is fed to every rule,
    so the comparison is apples-to-apples. Rules that can't run in the
    environment (e.g. archetype without the ``archetypes`` package) are skipped
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
            if hasattr(sel, "seed") and getattr(sel, "seed") is None:
                sel.seed = 0

            scores = []
            for q in query_ids:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        chosen = sel.select(matrix, q)
                    except (ImportError, NotImplementedError) as exc:
                        warnings.warn(f"Skipping selector '{name}': {exc}", stacklevel=2)
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
