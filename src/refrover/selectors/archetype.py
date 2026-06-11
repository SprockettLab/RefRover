import numpy as np
import pandas as pd
from .base import BaseSelector


class ArchetypeSelector(BaseSelector):
    """
    Archetype analysis in Jaccard similarity space.

    Treats each sample's row in the similarity submatrix as a feature vector,
    then finds the k archetypes (extreme points spanning the convex hull).
    Each archetype is mapped back to the nearest actual sample.

    Archetypes span the diversity space rather than summarising it, making
    them preferable to medoids for differential coverage: you want assemblies
    that together cover all the genomic variation present in the dataset, not
    assemblies that are typical representatives.

    Requires the `archetypes` package (pip install archetypes).
    """

    def __init__(self, k: int, min_jaccard: float = 0.1, random_state: int | None = 42):
        super().__init__(k, min_jaccard)
        self.random_state = random_state

    def select(self, sim_matrix: pd.DataFrame, query_id: str) -> list[str]:
        try:
            from archetypes import AA
        except ImportError as exc:
            raise ImportError(
                "ArchetypeSelector requires the 'archetypes' package: pip install archetypes"
            ) from exc

        candidates = self._candidates(sim_matrix, query_id)
        if len(candidates) <= self.k:
            return self._trim_to_k(candidates)

        X = sim_matrix.loc[candidates, candidates].to_numpy().astype(float)
        k = min(self.k, len(candidates))

        aa = AA(n_archetypes=k, random_state=self.random_state)
        aa.fit(X)

        # Map each archetype to its nearest candidate (unique assignment)
        selected, used = [], set()
        for arch in aa.archetypes_:
            dists = np.linalg.norm(X - arch, axis=1)
            for idx in np.argsort(dists):
                cid = candidates[idx]
                if cid not in used:
                    selected.append(cid)
                    used.add(cid)
                    break

        # Ensure query is included — if not, replace the archetype whose nearest
        # sample is closest to query
        if query_id not in selected:
            sim_row = sim_matrix.loc[query_id, selected].to_numpy()
            most_replaceable = selected[int(np.argmax(sim_row))]
            selected[selected.index(most_replaceable)] = query_id

        return selected
