"""Shared test fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from cosmic_discovery.models import DataSource, LightCurve
from cosmic_discovery.database import Database


@pytest.fixture
def db(tmp_path):
    """Create a fresh in-memory-like SQLite database for testing."""
    db_path = tmp_path / "test.db"
    return Database(db_path)


@pytest.fixture
def demo_lc():
    """Create a simple synthetic light curve with a known transit."""
    rng = np.random.default_rng(42)
    n = 1000
    time = np.linspace(0, 27.4, n)  # one TESS sector
    flux = np.ones(n)
    # Add a transit every 5 days
    period = 5.0
    duration = 0.1  # days (~2.4 hours)
    depth = 0.005  # 5000 ppm
    phase = time % period
    transit = phase < duration
    flux[transit] = 1.0 - depth
    # Add noise
    flux += rng.normal(0, 0.0002, n)
    flux_err = np.full(n, 0.0002)

    return LightCurve(
        source_id="TIC_TEST_001",
        time=time,
        flux=flux,
        flux_err=flux_err,
        source=DataSource.DEMO,
        cadence_seconds=120.0,
    )


@pytest.fixture
def flat_lc():
    """Create a flat light curve with no transit."""
    rng = np.random.default_rng(99)
    n = 500
    time = np.linspace(0, 10, n)
    flux = np.ones(n) + rng.normal(0, 0.0001, n)
    flux_err = np.full(n, 0.0001)

    return LightCurve(
        source_id="TIC_FLAT_001",
        time=time,
        flux=flux,
        flux_err=flux_err,
        source=DataSource.DEMO,
        cadence_seconds=120.0,
    )
