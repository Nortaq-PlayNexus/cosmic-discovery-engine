"""Synthetic light-curve primitives shared by benchmarks and diagnostics.

These are *fixtures*, not data sources. Anything produced here is synthetic and
must never be presented as an observation.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from ..models import DataSource, LightCurve

__all__ = ["flat_series", "inject_box", "flat_lightcurve", "inject_transit_into"]


def flat_series(
    n_points: int,
    cadence_minutes: float = 10.0,
    noise_ppm: float = 200.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """White-noise baseline as (time_days, flux).

    Caller is responsible for bounding ``n_points``; nothing here allocates
    based on the values passed in.
    """
    n = int(n_points)
    rng = np.random.default_rng(seed)
    time = np.arange(n) * (cadence_minutes / 1440.0)
    flux = 1.0 + rng.normal(0.0, noise_ppm * 1e-6, n)
    return time, flux


def inject_box(
    time: np.ndarray,
    flux: np.ndarray,
    period_days: float,
    depth_ppm: float,
    duration_days: float,
    epoch: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Add a box-shaped transit. Returns copies; inputs are not modified."""
    out = flux.copy()
    phase = ((time - epoch) % period_days) / period_days
    out[phase < (duration_days / period_days)] -= depth_ppm * 1e-6
    return time, out


def flat_lightcurve(
    source_id: str = "flat",
    n_points: int = 4000,
    noise_ppm: float = 200.0,
    seed: int = 0,
    cadence_minutes: float = 10.0,
) -> LightCurve:
    time, flux = flat_series(n_points, cadence_minutes, noise_ppm, seed)
    return LightCurve(
        source_id=source_id,
        time=time,
        flux=flux,
        flux_err=np.full(len(time), noise_ppm * 1e-6),
        source=DataSource.DEMO,
        cadence_seconds=cadence_minutes * 60,
        retrieved_at=datetime.now(UTC),
    )


def inject_transit_into(
    lc: LightCurve,
    period_days: float,
    depth_ppm: float,
    duration_days: float,
    epoch: float = 0.05,
) -> LightCurve:
    """Return a copy of ``lc`` with a transit added."""
    _, flux = inject_box(
        lc.time, lc.flux, period_days, depth_ppm, duration_days, epoch
    )
    return LightCurve(
        source_id=lc.source_id,
        time=lc.time.copy(),
        flux=flux,
        flux_err=lc.flux_err.copy(),
        source=lc.source,
        cadence_seconds=lc.cadence_seconds,
        retrieved_at=lc.retrieved_at,
        metadata=dict(lc.metadata),
    )