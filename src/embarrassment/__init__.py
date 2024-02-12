from importlib.metadata import version  # pragma: no cover

from .api import (
    clean,
    neighbor_attr_triples,
    neighbor_rel_triples,
    neighbor_set,
    search,
    select,
    select_by_type,
    select_rel,
)

__version__ = version(__package__)
__all__ = [
    "clean",
    "select",
    "select_by_type",
    "select_rel",
    "search",
    "neighbor_set",
    "neighbor_attr_triples",
    "neighbor_rel_triples",
]
