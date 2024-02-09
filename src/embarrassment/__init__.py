from importlib.metadata import version  # pragma: no cover

from .api import clean, search, select, select_by_type, select_rel

__version__ = version(__package__)
__all__ = ["clean", "select", "select_by_type", "select_rel", "search"]
