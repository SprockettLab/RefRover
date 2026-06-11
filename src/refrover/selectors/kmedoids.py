import numpy as np
import pandas as pd
from .base import BaseSelector


def _kmedoids(dist: np.ndarray, k: int, rng: np.random.Generator) -> list[int]:
    """
    PAM-style k-medoids on a precomputed distance matrix.
    Initialises with greedy furthest-first, then runs swap improvements.
    Returns indices of the k medoids.
    """
    n = len(dist)
    k = min(k, n)

    # Greedy furthest-first initialisation
    first = int(rng.integers(n))
    medoids = [first]
    while len(medoids) < k:
        min_dists = dist[:, medoids].min(axis=1)
        medoids.append(int(np.argmax(min_dists)))

    def total_cost(meds):
        return dist[:, meds].min(axis=1).sum()

    # Swap phase: try replacing each medoid with each non-medoid
    improved = True
    while improved:
        improved = False
        for i in range(len(medoids)):
            current_cost = total_cost(medoids)
            for cand in range(n):
                if cand in medoids:
                    continue
                trial = medoids.copy()
                trial[i] = cand
                if total_cost(trial) < current_cost - 1e-10:
                    medoids[i] = cand
                    improved = True
                    break
            if improved:
                break

    return medoids


class KMedoidsSelector(BaseSelector):
    """
    k-medoids clustering in Jaccard distance space; returns the k medoids.
    Medoids are real samples (not centroids), making them valid alignment
    references.
    """

    def __init__(self, k: int, min_jaccard: float = 0.1, random_state: int | None = 42):
        super().__init__(k, min_jaccard)
        self.random_state = random_state

    def select(self, sim_matrix: pd.DataFrame, query_id: str) -> list[str]:
        candidates = self._candidates(sim_matrix, query_id)
        if len(candidates) <= self.k:
            return self._trim_to_k(candidates)

        dist = 1.0 - sim_matrix.loc[candidates, candidates].to_numpy()
        np.fill_diagonal(dist, 0.0)

        rng = np.random.default_rng(self.random_state)
        medoid_indices = _kmedoids(dist, self.k, rng)

        # Ensure query is in the result — swap the medoid most similar to query
        # for query itself if query wasn't selected
        query_idx = candidates.index(query_id)
        if query_idx not in medoid_indices:
            sim_to_query = sim_matrix.loc[query_id, candidates].to_numpy()
            closest = min(medoid_indices, key=lambda i: -sim_to_query[i])
            medoid_indices[medoid_indices.index(closest)] = query_idx

        return [candidates[i] for i in medoid_indices]
