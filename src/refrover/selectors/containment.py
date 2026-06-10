"""
ContainmentSelector: prototype selection driven by cross-sample containment.

Unlike the other selectors, this one does NOT use the assembly-vs-assembly
Jaccard similarity matrix. Instead it uses a reads-vs-assembly containment
matrix (containment[i, j] = fraction of sample i's read k-mers found in
assembly j), which directly measures how well sample i's reads will map to
assembly j.

Two modes
---------
unweighted (default)
    Apply MaxMin greedy selection on the containment matrix, treating
    containment as the similarity measure. Selects prototypes that are
    maximally diverse in read-mapping space.

weighted (requires species_weights)
    Weight each candidate assembly j by its predicted differential coverage
    signal, estimated from per-species abundance variance. Assemblies whose
    dominant species vary across samples get higher weight.

    species_weights: pd.Series indexed by assembly_id, values in [0, 1].
    Can be derived from gtdb_species_stats.tsv (see explore_species.R):
      weight_j = mean CV of species whose primary assembly is j,
                 restricted to mid-abundance species.

Input matrix shape
------------------
The containment matrix has:
  - rows: read sample IDs (query samples)
  - columns: assembly IDs (prototype candidates)

Unlike BaseSelector which takes a square assembly-vs-assembly matrix,
ContainmentSelector takes a rectangular reads-vs-assemblies matrix.
The query_id must be a row index (read sample ID).
"""

import warnings
import pandas as pd
import numpy as np
from typing import Optional


class ContainmentSelector:
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
        Per-assembly weight Series (index = assembly_id). If provided,
        candidate scores are multiplied by weight[assembly_id].
    """

    def __init__(
        self,
        k: int,
        min_containment: float = 0.05,
        weights: Optional[pd.Series] = None,
    ):
        self.k = k
        self.min_containment = min_containment
        self.weights = weights

    def select(self, containment_matrix: pd.DataFrame, query_id: str) -> list[str]:
        """
        Select k prototype assemblies for query_id.

        Parameters
        ----------
        containment_matrix : pd.DataFrame
            Rows = read sample IDs, columns = assembly IDs.
            Values = containment(reads_i, assembly_j).
        query_id : str
            Row index of the query sample in containment_matrix.

        Returns
        -------
        list[str]
            Ordered list of prototype assembly IDs. The query sample's own
            assembly is always first (containment = 1.0 by definition).
        """
        if query_id not in containment_matrix.index:
            raise KeyError(f"query_id '{query_id}' not found in containment matrix rows")

        query_row = containment_matrix.loc[query_id]

        # Filter candidates by minimum containment
        candidates = query_row[query_row >= self.min_containment].index.tolist()

        # The query's own assembly should always be first if present
        own_assembly = query_id  # assumes assembly ID matches sample ID
        if own_assembly in candidates:
            candidates = [own_assembly] + [c for c in candidates if c != own_assembly]
        elif candidates:
            # Put the highest-containment assembly first
            candidates = sorted(candidates, key=lambda c: -float(query_row[c]))

        if len(candidates) == 0:
            raise ValueError(
                f"No assemblies meet min_containment={self.min_containment} for {query_id}"
            )

        if len(candidates) < self.k:
            warnings.warn(
                f"Only {len(candidates)} candidates available (k={self.k}). "
                "Returning all candidates.",
                UserWarning,
                stacklevel=2,
            )

        if len(candidates) <= self.k:
            return candidates

        # MaxMin greedy selection in containment space, with optional weighting
        selected = [candidates[0]]
        remaining = candidates[1:]

        # Track min containment-distance from selected set for each remaining candidate
        # distance = 1 - containment(assembly_j, assembly_k), approximated via
        # the cross-sample containment profiles: dist(j, k) = 1 - corr(col_j, col_k)
        all_rows = containment_matrix.values  # shape (n_samples, n_assemblies)
        col_index = {c: containment_matrix.columns.get_loc(c) for c in candidates}

        def _profile(assembly_id):
            return all_rows[:, col_index[assembly_id]]

        # Pairwise distance proxy: 1 - Pearson correlation between containment profiles
        min_dist = np.array([
            1.0 - float(np.corrcoef(_profile(selected[0]), _profile(c))[0, 1])
            for c in remaining
        ])
        min_dist = np.nan_to_num(min_dist, nan=0.0)

        while len(selected) < self.k and remaining:
            scores = min_dist.copy()

            # Apply species-level weights if provided
            if self.weights is not None:
                w = np.array([
                    float(self.weights.get(c, 1.0)) for c in remaining
                ])
                scores = scores * w

            best_idx = int(np.argmax(scores))
            best = remaining[best_idx]
            selected.append(best)

            # Update min distances
            new_dists = np.array([
                1.0 - float(np.corrcoef(_profile(best), _profile(c))[0, 1])
                for c in remaining
            ])
            new_dists = np.nan_to_num(new_dists, nan=0.0)
            min_dist = np.minimum(min_dist, new_dists)

            remaining.pop(best_idx)
            min_dist = np.delete(min_dist, best_idx)

        return selected
