from .base import BaseSelector
from .random import RandomSelector
from .maxmin import MaxMinSelector
from .kmedoids import KMedoidsSelector
from .archetype import ArchetypeSelector

SELECTOR_REGISTRY: dict[str, type[BaseSelector]] = {
    "random": RandomSelector,
    "maxmin": MaxMinSelector,
    "kmedoids": KMedoidsSelector,
    "archetype": ArchetypeSelector,
}

__all__ = [
    "BaseSelector",
    "RandomSelector",
    "MaxMinSelector",
    "KMedoidsSelector",
    "ArchetypeSelector",
    "SELECTOR_REGISTRY",
]
