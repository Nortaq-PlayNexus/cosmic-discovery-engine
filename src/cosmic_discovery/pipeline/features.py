"""Feature extraction from light curves: Lomb-Scargle periodogram and BLS transit search."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from astropy.timeseries import BoxLeastSquares, LombScargle

from ..models import LightCurve

logger = logging.getLogger(__name__)


@dataclass
class LightCurveFeatures:
    """Extracted features from a light curve."""

    source_id: str
    n_points: int
    baseline_days: float
    median_flux: float
    rms_ppm: float
    skewness: float
    kurtosis: float

    # Lomb-Scargle results
    ls_period_days: Optional[float] = None
    ls_power: float = 0.0
    ls_fap: float = 1.0

    # BLS results
    bls_period_days: Optional[float] = None
    bls_epoch_bjd: Optional[float] = None
    bls_duration_hours: Optional[float] = None
    bls_depth_ppm: Optional[float] = None
    bls_snr: float = 0.0
    bls_n_transits: int = 0

    # Transit shape
    even_odd_depth_ratio: Optional[float] = None
    duty_cycle: float = 0.0

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "n_points": self.n_points,
            "baseline_days": self.baseline_days,
            "median_flux": self.median_flux,
            "rms_ppm": self.rms_ppm,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "ls_period_days": self.ls_period_days,
            "ls_power": self.ls_power,
            "ls_fap": self.ls_fap,
            "bls_period_days": self.bls_period_days,
            "bls_epoch_bjd": self.bls_epoch_bjd,
            "bls_duration_hours": self.bls_duration_hours,
            "bls_depth_ppm": self.bls_depth_ppm,
            "bls_snr": self.bls_snr,
            "bls_n_transits": self.bls_n_transits,
            "even_odd_depth_ratio": self.even_odd_depth_ratio,
            "duty_cycle": self.duty_cycle,
        }


def extract_features(
    lc: LightCurve,
    min_period_days: float = 0.1,
    max_period_days: float = 50.0,
    bls_duration_grid: list[float] | None = None,
    detrend_window: int = 101,
    outlier_sigma: float = 5.0,
) -> LightCurveFeatures:
    """Extract all features from a light curve.

    Steps:
    1. Clean: remove NaNs, sigma-clip outliers, detrend.
    2. Lomb-Scargle periodogram for stellar variability.
    3. Box Least Squares for transit detection.
    4. Even/odd transit depth comparison.
    """
    # ── Clean ──────────────────────────────────────────────────────────
    time, flux, flux_err = _clean_lightcurve(lc, outlier_sigma=outlier_sigma)

    if len(time) < 50:
        logger.warning("Too few points after cleaning for %s: %d", lc.source_id, len(time))
        return LightCurveFeatures(
            source_id=lc.source_id,
            n_points=len(time),
           baseline_days=lc.baseline_days,
            median_flux=lc.median_flux,
            rms_ppm=lc.rms_ppm,
            skewness=0.0,
            kurtosis=0.0,
        )

    # ── Basic statistics ───────────────────────────────────────────────
    from scipy import stats

    skewness = float(stats.skew(flux))
    kurtosis = float(stats.kurtosis(flux))

    # ── Detrend (moving median) ────────────────────────────────────────
    # The window must be several times longer than the longest transit we mean
    # to detect, otherwise the trend absorbs the transit and divides it out.
    cadence_days = _median_cadence_days(time)
    max_durations = bls_duration_grid or [0.05, 0.1, 0.15, 0.2]
    max_transit_days = max(max_durations)
    required = required_detrend_window_points(cadence_days, max_transit_days)
    if detrend_window < required:
        logger.info(
            "Raising detrend window %d -> %d for %s: cadence %.4f d, "
            "longest transit %.3f d (window would be %.2fx the transit)",
            detrend_window, required, lc.source_id, cadence_days,
            max_transit_days, detrend_window * cadence_days / max_transit_days,
        )
        detrend_window = required

    detrended = _detrend(time, flux, window=detrend_window)

    # ── Lomb-Scargle ───────────────────────────────────────────────────
    ls_period, ls_power, ls_fap = _run_lomb_scargle(
        time, detrended, flux_err,
        min_period=min_period_days,
        max_period=max_period_days,
    )

    # ── BLS transit search ─────────────────────────────────────────────
    bls_results = _run_bls(
        time, detrended, flux_err,
        min_period=min_period_days,
        max_period=max_period_days,
        duration_grid=bls_duration_grid,
    )

    # ── Even/odd depth ─────────────────────────────────────────────────
    even_odd_ratio = None
    if bls_results is not None:
        even_odd_ratio = _even_odd_depth(time, detrended, bls_results)

    return LightCurveFeatures(
        source_id=lc.source_id,
        n_points=len(time),
        baseline_days=float(time[-1] - time[0]),
        median_flux=float(np.median(flux)),
        rms_ppm=float(np.std(detrended) / np.median(flux) * 1e6),
        skewness=skewness,
        kurtosis=kurtosis,
        ls_period_days=ls_period,
        ls_power=ls_power,
        ls_fap=ls_fap,
        bls_period_days=bls_results["period"] if bls_results else None,
        bls_epoch_bjd=bls_results["epoch"] if bls_results else None,
        bls_duration_hours=bls_results["duration"] * 24 if bls_results else None,
        bls_depth_ppm=bls_results["depth"] * 1e6 if bls_results else None,
        bls_snr=bls_results["snr"] if bls_results else 0.0,
        bls_n_transits=bls_results["n_transits"] if bls_results else 0,
        even_odd_depth_ratio=even_odd_ratio,
        duty_cycle=bls_results["duty_cycle"] if bls_results else 0.0,
    )


def required_detrend_window_points(
    cadence_days: float, max_transit_days: float, safety_factor: float = 12.0
) -> int:
    """Smallest detrend window that will not absorb a transit.

    A moving median shorter than the transit drags the transit into the trend
    and divides it out. Measured on synthetic injections (`scripts/
    characterize_defects.py`), recovery of a 1000 ppm / 6 h transit by
    window:transit ratio is 57% at x2, 88% at x4, 94% at x6, 96% at x8 and
    99% at x12. The shipped default was x4, which leaves ~12% of the depth
    unrecovered; x12 is the smallest factor tested that reaches ~99%.

    CAVEAT, established the same way and NOT solved by this function: the
    window is a trade-off between two independent failure modes. Too short
    absorbs the transit; too long absorbs stellar rotation, which is real
    astrophysical signal. Sweeping rotation periods from 0.7 to 12 d showed no
    constant factor is safe for both at once — x4 keeps only 2% of a 6 d
    rotation, x16 still loses 87%. Choosing the window from the measured
    variability period instead was tried and did not help, because the
    Lomb-Scargle peak used to pick it is itself distorted by the transit.
    Transit recovery therefore stays high at the cost of flattening slow
    rotation, and that trade is unresolved. Treat this as mitigation, not a fix.

    Expressed in points because the window is applied over samples; the
    cadence must be in days.
    """
    if cadence_days <= 0 or max_transit_days <= 0:
        return 101
    required = safety_factor * max_transit_days / cadence_days
    n = int(np.ceil(required))
    return n + 1 if n % 2 == 0 else n  # median_filter needs an odd size


def _clean_lightcurve(
    lc: LightCurve,
    outlier_sigma: float = 5.0,
    negative_sigma: float | None = None,
    max_removed_fraction: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove non-finite samples and clip outliers asymmetrically.

    The clip is one-sided on purpose. Instrumental contamination (scattered
    light, cosmic rays, pointing jumps) shows up as *positive* flux excursions,
    while a transit is a dip. Measured on the demo light curve, a symmetric
    5-sigma clip removed 76% of in-transit samples. Downward excursions are
    therefore left alone by default, because any fixed sigma threshold deep
    enough to catch contamination also deletes deep transits — a 100-sigma
    transit is not an outlier, it is the signal. Pass an explicit
    ``negative_sigma`` to clip the low side too.

    The scale is a MAD rather than a standard deviation so the transit itself
    cannot inflate the width used to judge it, and so that the upward threshold
    stays meaningful when a few samples carry enormous excursions.
    """
    mask = np.isfinite(lc.time) & np.isfinite(lc.flux) & np.isfinite(lc.flux_err)
    mask &= lc.flux > 0
    mask &= lc.flux_err > 0

    time = lc.time[mask]
    flux = lc.flux[mask]
    flux_err = lc.flux_err[mask]

    if len(time) < 10:
        return time, flux, flux_err

    if negative_sigma is None:
        negative_sigma = 0.0  # 0 means "do not clip downward"

    # Robust scale: MAD -> sigma. Immune to the transit and to bright outliers.
    med = np.median(flux)
    mad = np.median(np.abs(flux - med))
    robust_sigma = 1.4826 * mad

    if robust_sigma <= 0:
        # MAD degenerates to zero when many samples share identical values
        # (quantised, saturated or synthetic-flat data). Fall back to the
        # standard deviation, which is still far better than no clipping.
        robust_sigma = float(np.std(flux))
        logger.debug(
            "MAD scale is zero for %s; falling back to std=%.3e",
            lc.source_id, robust_sigma,
        )

    if robust_sigma > 0:
        resid = flux - med
        keep = resid < outlier_sigma * robust_sigma
        if negative_sigma and negative_sigma > 0:
            keep &= resid > -negative_sigma * robust_sigma
        clip = keep
        removed_fraction = 1.0 - clip.mean()
        # The guard exists to catch a degenerate robust scale, not to veto
        # ordinary clipping. Removing a few percent is exactly what clipping is
        # for; only a wholesale collapse indicates the threshold went wrong.
        if removed_fraction > max_removed_fraction:
            logger.warning(
                "Sigma-clip would remove %.0f%% of points (limit %.0f%%); "
                "skipping clip for %s",
                removed_fraction * 100, max_removed_fraction * 100, lc.source_id,
            )
        else:
            time, flux, flux_err = time[clip], flux[clip], flux_err[clip]

    return time, flux, flux_err


def _median_cadence_days(time: np.ndarray) -> float:
    """Median sample spacing in days."""
    if len(time) < 2:
        return 0.0
    return float(np.median(np.diff(np.sort(time))))


def _detrend(time: np.ndarray, flux: np.ndarray, window: int = 101) -> np.ndarray:
    """Detrend using a moving median filter."""
    from scipy.ndimage import median_filter

    if len(flux) < window:
        window = len(flux) // 2 * 2 + 1  # make odd
        if window < 3:
            return flux / np.median(flux)

    trend = median_filter(flux, size=window, mode="reflect")
    # Avoid division by zero
    trend = np.where(trend == 0, np.median(flux), trend)
    return flux / trend


def _run_lomb_scargle(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    min_period: float = 0.1,
    max_period: float = 50.0,
) -> tuple[Optional[float], float, float]:
    """Run Lomb-Scargle periodogram. Returns (best_period, power, FAP)."""
    try:
        ls = LombScargle(time, flux, dy=flux_err)
        frequency, power = ls.autopower(
            minimum_frequency=1.0 / max_period,
            maximum_frequency=1.0 / min_period,
            samples_per_peak=10,
        )
        if len(power) == 0:
            return None, 0.0, 1.0

        best_idx = np.argmax(power)
        best_period = 1.0 / frequency[best_idx]
        best_power = float(power[best_idx])

        # False alarm probability
        fap = float(ls.false_alarm_probability(best_power))

        return float(best_period), best_power, fap
    except Exception as e:
        logger.warning("Lomb-Scargle failed: %s", e)
        return None, 0.0, 1.0


def _run_bls(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    min_period: float = 0.1,
    max_period: float = 50.0,
    duration_grid: list[float] | None = None,
) -> Optional[dict[str, Any]]:
    """Run Box Least Squares transit search."""
    if duration_grid is None:
        duration_grid = [0.05, 0.1, 0.15, 0.2]

    # Ensure all durations are shorter than the minimum period
    duration_grid = [d for d in duration_grid if d < min_period]
    if not duration_grid:
        duration_grid = [min_period * 0.5]

    try:
        bls = BoxLeastSquares(time, flux, dy=flux_err)
        periods = np.linspace(min_period, max_period, 10000)
        durations = np.array(duration_grid)  # in days

        results = bls.power(periods, durations)
        if results is None or len(results.power) == 0:
            return None

        best_idx = np.argmax(results.power)
        best_period = float(results.period[best_idx])
        best_duration = float(results.duration[best_idx])
        best_epoch = float(results.transit_time[best_idx])
        best_depth = float(results.depth[best_idx])

        # Compute SNR from BLS depth and its uncertainty
        stats_dict = bls.compute_stats(best_period, best_duration, best_epoch)
        depth_tuple = stats_dict["depth"]
        # astropy returns depth as (depth_value, depth_error)
        if isinstance(depth_tuple, (tuple, list)) and len(depth_tuple) >= 2:
            best_depth = float(depth_tuple[0])
            depth_err = float(depth_tuple[1])
        else:
            best_depth = float(best_depth)
            depth_err = 0.0

        snr = best_depth / depth_err if depth_err > 0 else 0.0

        # Count transits
        n_transits = int((time[-1] - time[0]) / best_period)

        # Duty cycle
        duty_cycle = best_duration / best_period

        return {
            "period": best_period,
            "duration": best_duration,
            "epoch": best_epoch,
            "depth": best_depth,
            "snr": snr,
            "n_transits": n_transits,
            "duty_cycle": duty_cycle,
            "power": float(results.power[best_idx]),
        }
    except Exception as e:
        logger.warning("BLS failed: %s", e)
        return None


def _even_odd_depth(
    time: np.ndarray, flux: np.ndarray, bls_results: dict[str, Any]
) -> Optional[float]:
    """Compare depths of even-numbered vs odd-numbered transits.

    A significant difference may indicate an eclipsing binary
    (two stars with different radii) rather than a planet.
    """
    period = bls_results["period"]
    duration = bls_results["duration"]
    epoch = bls_results["epoch"]

    phases = ((time - epoch) % period) / period
    # Phase 0.5 = secondary eclipse
    in_transit = phases < (duration / period)
    in_secondary = (phases > 0.5) & (phases < 0.5 + duration / period)

    # Split transits into even and odd
    transit_numbers = np.floor((time - epoch) / period).astype(int)
    even_mask = in_transit & (transit_numbers % 2 == 0)
    odd_mask = in_transit & (transit_numbers % 2 == 1)

    if even_mask.sum() < 3 or odd_mask.sum() < 3:
        return None

    even_depth = 1.0 - np.median(flux[even_mask])
    odd_depth = 1.0 - np.median(flux[odd_mask])

    if odd_depth == 0:
        return None

    return float(even_depth / odd_depth)
