import pandas as pd
from .base import BaseSelector


class FeedbackSelector(BaseSelector):
    """
    Coverage-feedback selector: map reads → measure actual variance → iterate.

    Selects an initial set of prototypes, aligns reads, computes real per-contig
    coverage variance, then refines the selection to maximise observed variance.

    Most accurate selector; most expensive (requires alignment per iteration).
    Not yet implemented — depends on align.py + coverage.py being wired up.
    """

    def select(self, sim_matrix: pd.DataFrame, query_id: str) -> list[str]:
        raise NotImplementedError(
            "FeedbackSelector requires alignment + CoverM; not yet implemented. "
            "Use archetype or greedy_var instead."
        )
