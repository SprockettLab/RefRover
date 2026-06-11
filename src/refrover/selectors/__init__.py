from .base import BaseSelector
from .random import RandomSelector
from .maxmin import MaxMinSelector
from .kmedoids import KMedoidsSelector
from .archetype import ArchetypeSelector
from .greedy_var import GreedyVarSelector
from .containment import ContainmentSelector
from .feedback import FeedbackSelector

SELECTOR_REGISTRY: dict[str, type[BaseSelector]] = {
    "random": RandomSelector,
    "maxmin": MaxMinSelector,
    "kmedoids": KMedoidsSelector,
    "archetype": ArchetypeSelector,
    "greedy_var": GreedyVarSelector,
    "containment": ContainmentSelector,
    "feedback": FeedbackSelector,
}

# Selectors that operate on a reads-vs-assembly containment matrix instead of
# the assembly-vs-assembly Jaccard matrix. The CLI/pipeline use this to decide
# which matrix to feed the selector.
CONTAINMENT_SELECTORS: set[str] = {"containment"}

__all__ = [
    "BaseSelector",
    "RandomSelector",
    "MaxMinSelector",
    "KMedoidsSelector",
    "ArchetypeSelector",
    "GreedyVarSelector",
    "ContainmentSelector",
    "FeedbackSelector",
    "SELECTOR_REGISTRY",
    "CONTAINMENT_SELECTORS",
]
