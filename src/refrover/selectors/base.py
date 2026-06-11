from abc import ABC, abstractmethod
import warnings
import pandas as pd


class BaseSelector(ABC):
    def __init__(self, k: int, min_jaccard: float = 0.1):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if not 0.0 <= min_jaccard <= 1.0:
            raise ValueError(f"min_jaccard must be in [0, 1], got {min_jaccard}")
        self.k = k
        self.min_jaccard = min_jaccard

    @abstractmethod
    def select(self, sim_matrix: pd.DataFrame, query_id: str) -> list[str]:
        """
        Select up to k prototype IDs for query_id.

        sim_matrix: symmetric pairwise Jaccard similarity indexed by sample ID,
                    with 1.0 on the diagonal.
        query_id:   sample whose reads will be aligned to the returned prototypes.
        returns:    list of selected prototype IDs, length <= k.
        """

    def _candidates(self, sim_matrix: pd.DataFrame, query_id: str) -> list[str]:
        if query_id not in sim_matrix.index:
            raise KeyError(f"query_id '{query_id}' not found in similarity matrix")
        row = sim_matrix.loc[query_id]
        candidates = row[row >= self.min_jaccard].index.tolist()
        if not candidates:
            raise ValueError(
                f"No candidates for '{query_id}' with min_jaccard={self.min_jaccard}. "
                "Lower min_jaccard or check that the similarity matrix is correct."
            )
        return candidates

    def _trim_to_k(self, candidates: list[str]) -> list[str]:
        if len(candidates) < self.k:
            warnings.warn(
                f"Only {len(candidates)} candidates available (k={self.k}). "
                "Returning all candidates.",
                stacklevel=3,
            )
        return candidates[: self.k]
