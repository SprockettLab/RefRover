from .base import BaseSelector
from .random import RandomSelector
from .maxmin import MaxMinSelector
from .kmedoids import KMedoidsSelector
from .archetype import ArchetypeSelector
from .greedy_var import GreedyVarSelector
from .feedback import FeedbackSelector

SELECTOR_REGISTRY: dict[str, type[BaseSelector]] = {
    "random": RandomSelector,
    "maxmin": MaxMinSelector,
    "kmedoids": KMedoidsSelector,
    "archetype": ArchetypeSelector,
    "greedy_var": GreedyVarSelector,
    "feedback": FeedbackSelector,
}

__all__ = [
    "BaseSelector",
    "RandomSelector",
    "MaxMinSelector",
    "KMedoidsSelector",
    "ArchetypeSelector",
    "GreedyVarSelector",
    "FeedbackSelector",
    "SELECTOR_REGISTRY",
]
