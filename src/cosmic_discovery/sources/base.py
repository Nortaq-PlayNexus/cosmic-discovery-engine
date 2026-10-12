"""Abstract base class for data source adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from ..models import LightCurve, StellarSource


@dataclass
class SourceResult:
    """Standard result from any data source."""

    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class DataSourceBase(ABC):
    """Base class all data sources must implement."""

    name: str = "base"

    @abstractmethod
    def fetch_lightcurve(
        self, source_id: str, **kwargs: Any
    ) -> SourceResult:
        """Fetch a light curve for the given source."""
        ...

    @abstractmethod
    def search_targets(
        self, ra_deg: float, dec_deg: float, radius_arcsec: float, **kwargs: Any
    ) -> SourceResult:
        """Search for targets near a sky position."""
        ...

    def health_check(self) -> SourceResult:
        """Check if the source is reachable. Override in subclasses."""
        return SourceResult(success=True, metadata={"source": self.name})
