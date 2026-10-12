"""Offline demo data source — synthetic light curves for testing.

This source generates realistic synthetic TESS-like light curves with
configurable transit signals, stellar variability, and systematic noise.
It is NOT real observational data and is clearly flagged as such.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..models import DataSource, LightCurve, StellarSource
from .base import DataSourceBase, SourceResult

logger = logging.getLogger(__name__)

# Known demo targets with plausible TIC IDs and sky positions
DEMO_TARGETS = [
    ("TIC 100000001", 15.0, 45.0, 10.5),
    ("TIC 100000002", 30.0, -10.0, 11.2),
    ("TIC 100000003", 180.0, 5.0, 9.8),
    ("TIC 100000004", 250.0, -30.0, 12.1),
    ("TIC 100000005", 310.0, 60.0, 10.9),
    ("TIC 100000006", 45.0, 15.0, 8.5),
    ("TIC 100000007", 120.0, -45.0, 11.8),
    ("TIC 100000008", 200.0, 30.0, 10.2),
]


class DemoSource(DataSourceBase):
    """Generate synthetic light curves that mimic TESS 2-minute cadence data."""

    name = "demo"

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def health_check(self) -> SourceResult:
        return SourceResult(success=True, metadata={"source": self.name})

    def search_targets(
        self, ra_deg: float, dec_deg: float, radius_arcsec: float, **kwargs: Any
    ) -> SourceResult:
        """Return demo targets near the requested position."""
        radius_deg = radius_arcsec / 3600.0
        found = []
        for tic_id, tic_ra, tic_dec, mag in DEMO_TARGETS:
            # Simple box filter
            if abs(tic_ra - ra_deg) <= radius_deg and abs(tic_dec - dec_deg) <= radius_deg:
                found.append(
                    StellarSource(
                        source_id=tic_id,
                        ra_deg=tic_ra,
                        dec_deg=tic_dec,
                        magnitude=mag,
                        tic_id=int(tic_id.replace("TIC ", "")),
                        metadata={"demo": True},
                    )
                )
        return SourceResult(success=True, data=found, metadata={"count": len(found)})

    def get_source_info(self, source_id: str) -> StellarSource | None:
        """Return StellarSource info for a demo target."""
        for tic_id, ra, dec, mag in DEMO_TARGETS:
            if tic_id == source_id:
                return StellarSource(
                    source_id=tic_id,
                    ra_deg=ra,
                    dec_deg=dec,
                    magnitude=mag,
                    tic_id=int(tic_id.replace("TIC ", "")),
                    metadata={"demo": True},
                )
        return None

    def fetch_lightcurve(
        self, source_id: str, **kwargs: Any
    ) -> SourceResult:
        """Generate a synthetic light curve with a transit signal.

        Every demo target gets a periodic transit-like dip so the
        pipeline has something to find. The signal parameters vary
        per target to create diverse test cases.
        """
        try:
            lc = self._generate_synthetic(source_id, **kwargs)
            return SourceResult(success=True, data=lc)
        except Exception as e:
            logger.error("Demo light curve generation failed for %s: %s", source_id, e)
            return SourceResult(success=False, error=str(e))

    def _generate_synthetic(
        self,
        source_id: str,
        n_days: float = 27.4,
        cadence_minutes: float = 2.0,
        period_days: float = 5.0,
        transit_depth_ppm: float = 5000.0,
        transit_duration_hours: float = 3.0,
        noise_ppm: float = 200.0,
        variability_ppm: float = 500.0,
        **kwargs: Any,
    ) -> LightCurve:
        """Create a synthetic light curve with a box-shaped transit.

        The signal is: flat baseline + sinusoidal stellar variability
        + Gaussian white noise + periodic box-shaped transits.
        """
        n_points = int(n_days * 24 * 60 / cadence_minutes)
        time = np.arange(n_points) * (cadence_minutes / 1440.0)  # days

        # Seed from source_id for reproducibility
        src_hash = int(hashlib.sha256(source_id.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(src_hash)

        # Baseline flux (normalised to ~1.0)
        baseline = np.ones(n_points)

        # Stellar variability (sinusoidal + long-term trend)
        var_period = rng.uniform(3, 15)  # days
        variability = 1.0 + (variability_ppm * 1e-6) * np.sin(
            2 * np.pi * time / var_period + rng.uniform(0, 2 * np.pi)
        )
        variability += (variability_ppm * 0.3e-6) * np.sin(
            2 * np.pi * time / (var_period * 0.37 + 0.1) + rng.uniform(0, 2 * np.pi)
        )

        # Transits: box-shaped dips
        duration_days = transit_duration_hours / 24.0
        depth = transit_depth_ppm * 1e-6
        phase = time % period_days
        transit_mask = phase < duration_days
        transit_signal = np.where(transit_mask, 1.0 - depth, 1.0)

        # Add secondary eclipse (shallower, half-period offset)
        secondary_depth = depth * rng.uniform(0.05, 0.15)
        secondary_phase = (phase + period_days / 2) % period_days
        secondary_mask = secondary_phase < duration_days
        transit_signal = np.where(secondary_mask, 1.0 - secondary_depth, transit_signal)

        # Combine
        flux = baseline * variability * transit_signal

        # Gaussian white noise
        noise = rng.normal(0, noise_ppm * 1e-6, n_points)
        flux += noise

        # Flux error estimate
        flux_err = np.full(n_points, noise_ppm * 1e-6)

        checksum = hashlib.sha256(
            time.tobytes() + flux.tobytes()
        ).hexdigest()[:16]

        return LightCurve(
            source_id=source_id,
            time=time,
            flux=flux,
            flux_err=flux_err,
            source=DataSource.DEMO,
            sector=kwargs.get("sector"),
            cadence_seconds=cadence_minutes * 60,
            retrieved_at=datetime.now(UTC),
            checksum=checksum,
            metadata={
                "synthetic": True,
                "period_days": period_days,
                "transit_depth_ppm": transit_depth_ppm,
                "transit_duration_hours": transit_duration_hours,
                "variability_ppm": variability_ppm,
                "noise_ppm": noise_ppm,
            },
        )
