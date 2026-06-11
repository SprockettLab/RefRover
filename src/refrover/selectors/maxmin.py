import numpy as np
import pandas as pd
from .base import BaseSelector


class MaxMinSelector(BaseSelector):
    """
    Greedy MaxMin: iteratively selects the candidate furthest from all
    already-selected prototypes, maximising the minimum pairwise distance
    in Jaccard space.

    Initialises with the query's own assembly (Jaccard = 1.0 to itself),
    so the query is always the first prototype.
    """

    def select(self, sim_matrix: pd.DataFrame, query_id: str) -> list[str]:
        candidates = self._candidates(sim_matrix, query_id)
        if len(candidates) <= self.k:
            return self._trim_to_k(candidates)

        dist = 1.0 - sim_matrix.loc[candidates, candidates].to_numpy()
        n = len(candidates)
        id_to_idx = {sid: i for i, sid in enumerate(candidates)}

        # Seed with query itself
        selected = [id_to_idx[query_id]]
        remaining = list(range(n))
        remaining.remove(id_to_idx[query_id])

        # min_dist_to_selected[i] = distance from candidate i to its nearest selected prototype
        min_dist = dist[np.array(remaining), :][:, selected].min(axis=1)

        while len(selected) < self.k and remaining:
            best_pos = int(np.argmax(min_dist))
            best_idx = remaining[best_pos]
            selected.append(best_idx)
            remaining.pop(best_pos)
            min_dist = np.delete(min_dist, best_pos)

            if remaining:
                new_dists = dist[np.array(remaining), best_idx]
                min_dist = np.minimum(min_dist, new_dists)

        return [candidates[i] for i in selected]
