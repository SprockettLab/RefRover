"""
ContainmentSelector: prototype selection driven by cross-sample containment.

Unlike the Jaccard-based selectors, this one operates on a reads-vs-assembly
*containment* matrix rather than an assembly-vs-assembly Jaccard matrix.
containment[i, j] = fraction of sample i's read k-mers found in assembly j —
a direct measure of how well sample i's reads will map to assembly j, which is
the quantity differential-coverage binning actually cares about.

Identifier model
----------------
A prototype is identified by the sample_id of the sample whose assembly it is.
The containment matrix is therefore square and labelled by sample_id on both
axes: rows are query (read) samples, columns are candidate assemblies. The
manifest resolves each returned sample_id to its assembly FASTA at align time
(see refrover.align._resolve_fastas). This is the same contract every other
selector follows, which lets the pipeline treat all selectors polymorphically.

Two modes
---------
unweighted (default)
    MaxMin greedy on containment-profile distance (1 - Pearson correlation of
    the two assemblies' cross-sample containment columns). Selects prototypes
    that are maximally diverse in read-mapping space.

weighted (requires `weights`)
    Each candidate's MaxMin score is multiplied by a per-assembly weight
    (a pd.Series indexed by sample_id). Use this to up-weight assemblies
    predicted to carry more differential signal.
"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from .base import BaseSelector


class ContainmentSelector(BaseSelector):
    """
    Prototype selection using cross-sample read containment.

    Parameters
    ----------
    k : int
        Number of prototypes to select.
    min_containment : float
        Minimum containment(query_reads, prototype_assembly) required.
        Assemblies below this threshold are excluded — reads won't map well.
    weights : pd.Series or None
        Per-assembly weight Series (index = sample_id). If provided, candidate
        MaxMin scores are multiplied by weight[sample_id] (default 1.0 for any
        assembly not in the Series).
    min_jaccard : float or None
        Alias accepted so the selector can be constructed uniformly from
        SELECTOR_REGISTRY with a `min_jaccard=` keyword. When given it overrides
        `min_containment` (the threshold semantics are identical: a minimum
        similarity floor on the query row).
    """

    def __init__(
        self,
        k: int,
        min_containment: float = 0.05,
        weights: Optional[pd.Series] = None,
        min_jaccard: Optional[float] = None,
    ):
        threshold = min_containment if min_jaccard is None else min_jaccard
        super().__init__(k, min_jaccard=threshold)
        # Expose under the domain-appropriate name as well.
        self.min_containment = threshold
        self.weights = weights

    def select(self, containment_matrix: pd.DataFrame, query_id: str) -> list[str]:
        """
        Select up to k prototype sample_ids for query_id.

        Parameters
        ----------
        containment_matrix : pd.DataFrame
            Square, labelled by sample_id on both axes. Rows = query (read)
            samples, columns = candidate assemblies. Values = containment.
        query_id : str
            Row index of the query sample.

        Returns
        -------
        list[str]
            Ordered prototype sample_ids, length <= k. The query's own assembly
            is first when it clears the threshold (containment 1.0 by definition).
        """
        if query_id not in containment_matrix.index:
            raise KeyError(f"query_id '{query_id}' not found in containment matrix rows")

        query_row = containment_matrix.loc[query_id]
        candidates = query_row[query_row >= self.min_containment].index.tolist()
        if not candidates:
            raise ValueError(
                f"No assemblies meet min_containment={self.min_containment} for "
                f"'{query_id}'. Lower min_containment or check the containment matrix."
            )

        # A prototype is identified by sample_id; the query's own assembly is the
        # column labelled query_id. Pin it first when present.
        if query_id in candidates:
            candidates = [query_id] + [c for c in candidates if c != query_id]
        else:
            candidates = sorted(candidates, key=lambda c: -float(query_row[c]))

        if len(candidates) < self.k:
            warnings.warn(
                f"Only {len(candidates)} candidates available (k={self.k}). "
                "Returning all candidates.",
                UserWarning,
                stacklevel=2,
            )
        if len(candidates) <= self.k:
            return candidates

        # Precompute the candidate-vs-candidate containment-profile distance ONCE.
        # dist(j, k) = 1 - Pearson correlation of the two assemblies' columns.
        corr = containment_matrix[candidates].corr().to_numpy()
        dist = 1.0 - np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(dist, 0.0)

        weights = None
        if self.weights is not None:
            weights = np.array([float(self.weights.get(c, 1.0)) for c in candidates])

        # MaxMin greedy, seeded with the query's own assembly (index 0).
        selected = [0]
        remaining = list(range(1, len(candidates)))
        min_dist = dist[0, remaining].copy()

        while len(selected) < self.k and remaining:
            scores = min_dist if weights is None else min_dist * weights[remaining]
            best_pos = int(np.argmax(scores))
            best = remaining[best_pos]
            selected.append(best)
            remaining.pop(best_pos)
            min_dist = np.delete(min_dist, best_pos)
            if remaining:
                min_dist = np.minimum(min_dist, dist[best, remaining])

        return [candidates[i] for i in selected]
