"""Tests for the demo data source."""

from __future__ import annotations

import numpy as np
import pytest

from cosmic_discovery.sources.demo import DemoSource, DEMO_TARGETS


def test_demo_search_returns_targets():
    source = DemoSource()
    result = source.search_targets(15.0, 45.0, 600.0)
    assert result.success
    assert result.data is not None
    assert len(result.data) > 0


def test_demo_search_empty_region():
    source = DemoSource()
    result = source.search_targets(0.0, -89.0, 10.0)
    assert result.success
    assert len(result.data) == 0


def test_demo_fetch_lightcurve():
    source = DemoSource()
    result = source.fetch_lightcurve("TIC 100000001")
    assert result.success
    lc = result.data
    assert lc.source_id == "TIC 100000001"
    assert lc.n_points > 100
    assert np.all(np.isfinite(lc.time))
    assert np.all(np.isfinite(lc.flux))


def test_demo_lightcurve_has_transit_signal():
    """Demo light curves should have detectable transit signals."""
    source = DemoSource()
    result = source.fetch_lightcurve("TIC 100000001")
    lc = result.data
    # Flux should dip below baseline during transits
    baseline = np.median(lc.flux)
    min_flux = np.min(lc.flux)
    assert min_flux < baseline * 0.995  # at least 0.5% dip


def test_demo_lightcurve_metadata():
    source = DemoSource()
    result = source.fetch_lightcurve("TIC 100000002")
    lc = result.data
    assert lc.metadata.get("synthetic") is True
    assert "period_days" in lc.metadata
    assert "transit_depth_ppm" in lc.metadata


def test_demo_all_targets_have_valid_ids():
    for tic_id, ra, dec, mag in DEMO_TARGETS:
        assert tic_id.startswith("TIC ")
        assert 0 <= ra < 360
        assert -90 <= dec <= 90
        assert mag > 0
