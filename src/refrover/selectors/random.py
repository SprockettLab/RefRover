import random
import pandas as pd
from .base import BaseSelector


class RandomSelector(BaseSelector):
    """
    Baseline: the query's own assembly plus random prototypes within threshold.

    Like the other selectors, the query is always the first prototype (it is the
    reference its own reads are guaranteed to map to), which keeps random a fair
    control: every selector is handed the same guaranteed anchor and differs only
    in how it picks the remaining k-1.
    """

    def __init__(self, k: int, min_jaccard: float = 0.1, seed: int | None = None):
        super().__init__(k, min_jaccard)
        self.seed = seed

    def select(self, sim_matrix: pd.DataFrame, query_id: str) -> list[str]:
        candidates = self._candidates(sim_matrix, query_id)
        if len(candidates) <= self.k:
            return self._trim_to_k(candidates)

        rng = random.Random(self.seed)
        if query_id in candidates:
            others = [c for c in candidates if c != query_id]
            return [query_id] + rng.sample(others, self.k - 1)
        return rng.sample(candidates, self.k)
