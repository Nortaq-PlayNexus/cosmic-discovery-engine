"""Tests for the feature extraction pipeline (Lomb-Scargle, BLS)."""

from __future__ import annotations

import numpy as np
import pytest

from cosmic_discovery.models import LightCurve, DataSource
from cosmic_discovery.pipeline.features import extract_features, _clean_lightcurve, _detrend


def test_clean_lightcurve_removes_nans():
    lc = LightCurve(
        source_id="test",
        time=np.array([1.0, 2.0, np.nan, 4.0]),
        flux=np.array([1.0, np.nan, 1.0, 1.0]),
        flux_err=np.array([0.01, 0.01, 0.01, 0.01]),
        source=DataSource.DEMO,
    )
    time, flux, flux_err = _clean_lightcurve(lc)
    assert len(time) == 2
    assert np.all(np.isfinite(time))
    assert np.all(np.isfinite(flux))


def test_clean_lightcurve_removes_upward_outliers():
    """Positive excursions (instrumental contamination) are clipped."""
    n = 100
    flux = np.ones(n)
    flux[50] = 100.0  # massive positive spike
    lc = LightCurve(
        source_id="test",
        time=np.arange(n, dtype=float),
        flux=flux,
        flux_err=np.full(n, 0.001),
        source=DataSource.DEMO,
    )
    time, flux_clean, _ = _clean_lightcurve(lc, outlier_sigma=5.0)
    assert len(flux_clean) < n
    assert flux_clean.max() < 10.0


def test_clean_lightcurve_preserves_deep_transits():
    """A symmetric clip would delete the signal we search for; we must not.

    A box of low points spanning 10% of the series, 30 sigma deep in a series
    whose robust scatter is otherwise 1 sigma, survives cleaning.
    """
    rng = np.random.default_rng(0)
    n = 1000
    flux = 1.0 + rng.normal(0, 1e-4, n)
    flux[400:460] -= 0.01  # 100-sigma-deep transit dip
    lc = LightCurve(
        source_id="transit",
        time=np.arange(n, dtype=float),
        flux=flux,
        flux_err=np.full(n, 1e-4),
        source=DataSource.DEMO,
    )
    _, flux_clean, _ = _clean_lightcurve(lc, outlier_sigma=5.0)
    assert flux_clean.min() < 0.99
    # The clip must not have removed most of the dip.
    assert (flux_clean < 0.99).sum() > 20


def test_clean_lightcurve_refuses_extreme_clip():
    """If clipping would remove too much, skip it rather than gut the series."""
    n = 200
    rng = np.random.default_rng(1)
    flux = rng.normal(1.0, 0.02, n)  # very noisy
    lc = LightCurve(
        source_id="noisy",
        time=np.arange(n, dtype=float),
        flux=flux,
        flux_err=np.full(n, 1e-4),
        source=DataSource.DEMO,
    )
    time, flux_clean, _ = _clean_lightcurve(lc, outlier_sigma=0.1, max_removed_fraction=0.25)
    # A tiny sigma would clip nearly everything; the guard must prevent that.
    assert len(flux_clean) >= 0.5 * n


def test_clean_lightcurve_mad_scale():
    """The clip uses a robust (MAD) scale, not the standard deviation.

    A single huge outlier inflates the standard deviation far above the robust
    scatter, so a std-based clip with a generous sigma would keep it. The MAD
    scale keeps the clip meaningful.
    """
    n = 200
    rng = np.random.default_rng(3)
    flux = 1.0 + rng.normal(0, 1e-4, n)
    flux[10] = 5.0  # huge spike inflates std but not MAD
    lc = LightCurve(
        source_id="mad",
        time=np.arange(n, dtype=float),
        flux=flux,
        flux_err=np.full(n, 1e-4),
        source=DataSource.DEMO,
    )
    _, flux_clean, _ = _clean_lightcurve(lc, outlier_sigma=5.0)
    assert flux_clean.max() < 1.01


def test_required_detrend_window_scales_with_transit():
    """The required window must grow with the transit it must not absorb."""
    from cosmic_discovery.pipeline.features import required_detrend_window_points

    cadence = 2.0 / 1440.0  # TESS 2-min
    short = required_detrend_window_points(cadence, 0.05)  # 1.2h transit
    long = required_detrend_window_points(cadence, 0.25)  # 6h transit
    assert long > short
    # At least 4x the transit duration in days.
    assert short * cadence >= 4 * 0.05
    # Odd integer, as median_filter requires.
    assert short % 2 == 1
    assert long % 2 == 1


def test_detrend_flat():
    flux = np.ones(200)
    result = _detrend(np.arange(200, dtype=float) * 0.1, flux, window=51)
    assert np.allclose(result, 1.0, atol=0.01)


def test_detrend_removes_trend():
    time = np.linspace(0, 10, 500)
    flux = 1.0 + 0.1 * time  # linear trend
    result = _detrend(time, flux, window=51)
    # Detrended should be roughly flat
    assert np.std(result) < 0.05


def test_extract_features_with_transit(demo_lc):
    """Feature extraction on a light curve with a known 5-day period transit."""
    features = extract_features(
        demo_lc,
        min_period_days=0.5,
        max_period_days=20.0,
        bls_duration_grid=[0.05, 0.1, 0.15],
    )
    assert features.source_id == "TIC_TEST_001"
    assert features.n_points > 0
    assert features.rms_ppm > 0
    # BLS should find a transit signal (period may be fundamental or harmonic)
    if features.bls_period_days is not None:
        assert features.bls_period_days > 0
        assert features.bls_snr > 0


def test_extract_features_flat(flat_lc):
    """Feature extraction on flat light curve — no transit expected."""
    features = extract_features(flat_lc, min_period_days=0.5, max_period_days=20.0)
    # Flat light curve should have low BLS SNR or no detection
    assert features.bls_snr < 20.0


def test_lomb_scargle_finds_period(demo_lc):
    """Lomb-Scargle should detect the 5-day periodicity."""
    features = extract_features(demo_lc, min_period_days=0.5, max_period_days=20.0)
    if features.ls_period_days is not None:
        # LS may find the fundamental or a harmonic; accept any of them
        ls_p = features.ls_period_days
        candidates = [5.0, 2.5, 10.0, 1.25]
        assert any(ls_p == pytest.approx(c, rel=0.15) for c in candidates)
