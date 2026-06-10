import random
import pandas as pd
from .base import BaseSelector


class RandomSelector(BaseSelector):
    """Baseline: random k prototypes within Jaccard threshold."""

    def __init__(self, k: int, min_jaccard: float = 0.1, seed: int | None = None):
        super().__init__(k, min_jaccard)
        self.seed = seed

    def select(self, sim_matrix: pd.DataFrame, query_id: str) -> list[str]:
        candidates = self._candidates(sim_matrix, query_id)
        if len(candidates) <= self.k:
            return self._trim_to_k(candidates)
        rng = random.Random(self.seed)
        return rng.sample(candidates, self.k)
