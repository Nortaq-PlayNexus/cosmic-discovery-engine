"""NASA Exoplanet Archive data source via TAP."""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
from astropy.table import Table

from ..models import DataSource, LightCurve, StellarSource, CatalogMatch
from .base import DataSourceBase, SourceResult

logger = logging.getLogger(__name__)


class ExoplanetArchiveSource(DataSourceBase):
    """Query the NASA Exoplanet Archive TAP service for known exoplanet data."""

    name = "exoplanet_archive"

    TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

    def __init__(
        self,
        tap_url: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 3.0,
    ) -> None:
        self.tap_url = tap_url or self.TAP_URL
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def health_check(self) -> SourceResult:
        try:
            self._query_tap("SELECT TOP 1 * FROM ps")
            return SourceResult(success=True, metadata={"source": self.name})
        except Exception as e:
            return SourceResult(success=False, error=str(e), metadata={"source": self.name})

    def search_targets(
        self, ra_deg: float, dec_deg: float, radius_arcsec: float, **kwargs: Any
    ) -> SourceResult:
        """Search for known exoplanet hosts near a position."""
        try:
            radius_deg = radius_arcsec / 3600.0
            # Use the Planetary Systems Composite table with spatial constraint
            query = f"""
                SELECT DISTINCT pl_name, ra, dec, sy_dist, pl_orbper,
                        pl_tranmid, pl_trandep, disc_year
                FROM ps
                WHERE 1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra_deg}, {dec_deg}, {radius_deg}))
                LIMIT 100
            """
            table = self._query_tap(query)
            if table is None or len(table) == 0:
                return SourceResult(success=True, data=[], metadata={"count": 0})

            matches = []
            for row in table:
                matches.append(
                    CatalogMatch(
                        catalogue="exoplanet_archive",
                        name=str(row["pl_name"]),
                        ra_deg=float(row["ra"]),
                        dec_deg=float(row["dec"]),
                        period_days=float(row["pl_orbper"]) if row["pl_orbper"] else None,
                        epoch_bjd=float(row["pl_tranmid"]) if row["pl_tranmid"] else None,
                        depth_ppm=float(row["pl_trandep"]) if row["pl_trandep"] else None,
                        metadata={
                            "sy_dist_pc": float(row["sy_dist"]) if row["sy_dist"] else None,
                            "discovery_year": int(row["disc_year"]) if row["disc_year"] else None,
                        },
                    )
                )
            return SourceResult(success=True, data=matches, metadata={"count": len(matches)})
        except Exception as e:
            logger.error("Exoplanet Archive search failed: %s", e)
            return SourceResult(success=False, error=str(e))

    def fetch_lightcurve(
        self, source_id: str, **kwargs: Any
    ) -> SourceResult:
        """Not implemented — Exoplanet Archive does not serve light curves."""
        return SourceResult(
            success=False,
            error="Exoplanet Archive does not serve light curves; use MAST instead",
        )

    def get_known_planets_for_crossmatch(
        self, ra_deg: float, dec_deg: float, radius_arcsec: float = 10.0
    ) -> list[CatalogMatch]:
        """Convenience wrapper for cross-matching."""
        result = self.search_targets(ra_deg, dec_deg, radius_arcsec)
        if result.success and result.data:
            return result.data
        return []

    def _query_tap(self, query: str) -> Optional[Table]:
        """Execute a TAP query with retries."""
        from astroquery.utils.tap.core import TapPlus

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                tap = TapPlus(url=self.tap_url)
                job = tap.launch_job(query)
                return job.get_results()
            except Exception as e:
                last_err = e
                logger.warning(
                    "TAP query attempt %d/%d failed: %s",
                    attempt + 1, self.max_retries, e,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        raise last_err or RuntimeError("TAP query failed")
