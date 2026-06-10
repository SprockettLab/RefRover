import pandas as pd
from .base import BaseSelector


class GreedyVarSelector(BaseSelector):
    """
    Greedy coverage-variance maximization.

    For each candidate assembly j, estimates the marginal contribution to
    cross-sample coverage variance using two factors:
      1. var_score: variance of j's similarity to all other samples — proxy
         for how much j's coverage will differ across samples.
      2. diversity: minimum distance from j to already-selected prototypes —
         penalizes adding a prototype redundant with one already chosen.

    Score = var_score * diversity. Query is always the first prototype.
    """

    def select(self, sim_matrix: pd.DataFrame, query_id: str) -> list[str]:
        candidates = self._candidates(sim_matrix, query_id)
        if len(candidates) <= self.k:
            return self._trim_to_k(candidates)

        all_ids = sim_matrix.columns.tolist()
        selected = [query_id]
        remaining = [c for c in candidates if c != query_id]

        # Pre-compute per-candidate variance scores (these don't change)
        var_scores = {c: float(sim_matrix.loc[c, all_ids].var()) for c in remaining}

        while len(selected) < self.k and remaining:
            best, best_score = None, -1.0
            for c in remaining:
                min_dist = min(1.0 - float(sim_matrix.loc[c, s]) for s in selected)
                score = var_scores[c] * min_dist
                if score > best_score:
                    best_score = score
                    best = c
            selected.append(best)
            remaining.remove(best)

        return selected
