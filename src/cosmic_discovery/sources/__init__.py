"""Data source adapters for Cosmic Discovery Engine."""

from .base import DataSourceBase, SourceResult
from .mast import MASTSource
from .exoplanet_archive import ExoplanetArchiveSource
from .demo import DemoSource

__all__ = ["DataSourceBase", "SourceResult", "MASTSource", "ExoplanetArchiveSource", "DemoSource"]
