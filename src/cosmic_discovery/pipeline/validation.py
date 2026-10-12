"""Statistical validation for transit candidates.

Three independent checks, deliberately built on different principles so that a
shared failure mode cannot make all three agree:

1. ``empirical_null`` — replicate the full BLS selection on phase-shuffled
   light curves to build a null distribution of the test statistic.
2. ``segment_holdout`` — detect on one half of the light curve, look for the
   signal in the *other* half, which was never seen.
3. ``injection_benchmark`` — inject signals of known parameters into flat noise
   and measure whether the pipeline recovers them. This is the positive
   control: a pipeline that cannot recover a strong injected signal cannot be
   trusted to report a null.

A fourth check, ``fdr_correction``, controls the family-wise false discovery
rate across candidates and is applied after the per-candidate null.

Applicability notes
-------------------
* The empirical p-value floor is ``1 / (1 + n_trials)``. With the default
  20 trials that is 0.0476 — small p-values are simply not resolvable at that
  trial count. This is reported alongside every p-value, never hidden.
* The null is computed on a coarser period grid than the science run to keep
  runtime bounded. That makes the null *slightly* optimistic in the extreme
  tail; the grid ratio is reported so the bias can be judged.
* BH assumes independence or positive dependence. BLS periods drawn from one
  light curve are arbitrarily dependent, so BY is the default. Both are
  computed; the caller decides which to act on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, Sequence

import numpy as np

from ..models import LightCurve
from .features import _clean_lightcurve, _detrend

logger = logging.getLogger(__name__)

# Precondition bounds for the FDR procedures.
_MIN_ALPHA = 1e-12
_MAX_ALPHA = 1.0


# ── FDR correction ────────────────────────────────────────────────────────


@dataclass
class FDRResult:
    """Outcome of a multiple-hypothesis correction over a candidate family."""

    method: str
    alpha: float
    n_tests: int
    rejected: list[str]
    n_rejected: int
    thresholds: list[float]
    applicability_ok: bool
    applicability_note: str

    @property
    def survives(self) -> bool:
        return self.n_rejected > 0


def _check_fdr_preconditions(
    p_values: Sequence[float], alpha: float
) -> tuple[bool, str]:
    """Verify the preconditions of the step-up FDR procedures.

    Returns (ok, note). The caller must not trust a corrected result when this
    returns False; the note says which precondition failed.
    """
    p = np.asarray(p_values, dtype=float)
    if p.size == 0:
        return False, "no tests supplied; correction undefined"
    if not np.all(np.isfinite(p)):
        return False, "non-finite p-values present"
    if np.any(p < 0.0) or np.any(p > 1.0):
        return False, "p-values outside [0, 1]"
    if not (_MIN_ALPHA <= alpha <= _MAX_ALPHA):
        return False, f"alpha={alpha} outside ({_MIN_ALPHA}, {_MAX_ALPHA}]"
    if np.all(p == p[0]):
        return True, "all p-values identical; correction degenerate but well-defined"
    return True, "preconditions satisfied"


def _step_up(
    p_values: Sequence[float], alpha: float, harmonic: float
) -> tuple[list[str], list[float], int]:
    """Shared step-up core for BH (harmonic=1) and BY (harmonic=C(m)).

    Returns (rejected_ids, thresholds, largest_k).
    """
    p = np.asarray(p_values, dtype=float)
    m = p.size
    order = np.argsort(p, kind="stable")
    scaled = alpha * (np.arange(1, m + 1) / m) / harmonic
    thresholds = scaled.tolist()

    # Largest k where p_(k) <= threshold_k. Ties handled by the stable sort.
    passed = np.nonzero(p[order] <= scaled)[0]
    k = int(passed.max()) + 1 if passed.size else 0
    rejected_idx = order[:k]
    return (
        [str(i) for i in rejected_idx],
        thresholds,
        k,
    )


def benjamini_hochberg(
    p_values: Sequence[float],
    labels: Sequence[str],
    alpha: float = 0.05,
) -> FDRResult:
    """Benjamini-Hochberg FDR control. Assumes independence or PRDS."""
    ok, note = _check_fdr_preconditions(p_values, alpha)
    m = len(p_values)
    if not ok:
        return FDRResult("BH", alpha, m, [], 0, [], False, note)

    rejected_idx, thresholds, k = _step_up(p_values, alpha, harmonic=1.0)
    rejected = [labels[int(i)] for i in rejected_idx]
    return FDRResult("BH", alpha, m, rejected, k, thresholds, True, note)


def benjamini_yekutieli(
    p_values: Sequence[float],
    labels: Sequence[str],
    alpha: float = 0.05,
) -> FDRResult:
    """Benjamini-Yekutieli FDR control. Valid under arbitrary dependence.

    This is the correct choice when the tests share a light curve, since BLS
    periods from one series are not independent.
    """
    ok, note = _check_fdr_preconditions(p_values, alpha)
    m = len(p_values)
    if not ok:
        return FDRResult("BY", alpha, m, [], 0, [], False, note)

    harmonic = float(np.sum(1.0 / np.arange(1, m + 1)))
    rejected_idx, thresholds, k = _step_up(p_values, alpha, harmonic=harmonic)
    rejected = [labels[int(i)] for i in rejected_idx]
    return FDRResult(
        "BY", alpha, m, rejected, k, thresholds, True,
        note + f"; harmonic C(m)={harmonic:.4f} for m={m}",
    )


# ── Empirical null from phase shuffling ───────────────────────────────────


@dataclass
class NullDistribution:
    """Monte Carlo null for the BLS selection statistic."""

    observed_snr: float
    n_trials: int
    null_snr: np.ndarray
    p_value: float
    p_value_floor: float
    n_exceeding: int
    grid_ratio: float
    seed: int
    method: str = "phase-shuffle BLS selection"
    note: str = ""

    @property
    def significant_at(self) -> bool:
        return self.p_value < 0.05

    def to_dict(self) -> dict[str, Any]:
        return {
            "observed_snr": self.observed_snr,
            "n_trials": self.n_trials,
            "p_value": self.p_value,
            "p_value_floor": self.p_value_floor,
            "n_exceeding": self.n_exceeding,
            "grid_ratio": self.grid_ratio,
            "seed": self.seed,
            "method": self.method,
            "note": self.note,
        }


def _bls_peak_snr(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    periods: np.ndarray,
    durations: np.ndarray,
) -> float:
    """Peak SNR of a BLS run. Returns -inf when the search is impossible."""
    from astropy.timeseries import BoxLeastSquares

    if len(periods) == 0 or len(durations) == 0:
        return float("-inf")
    keep = durations < periods.min()
    if not np.any(keep):
        return float("-inf")
    durations = durations[keep]

    bls = BoxLeastSquares(time, flux, dy=flux_err)
    power = bls.power(periods, durations)
    stats = bls.compute_stats(
        float(power.period[int(np.argmax(power.power))]),
        float(power.duration[int(np.argmax(power.power))]),
        float(power.transit_time[int(np.argmax(power.power))]),
    )
    depth_tuple = stats["depth"]
    if isinstance(depth_tuple, (tuple, list)) and len(depth_tuple) >= 2:
        depth, depth_err = float(depth_tuple[0]), float(depth_tuple[1])
        return depth / depth_err if depth_err > 0 else 0.0
    return 0.0


def _signal_free_residuals(
    detrended: np.ndarray, mask_sigma: float = 4.0
) -> tuple[np.ndarray, float]:
    """Residuals with the transit masked out, returned with the threshold used.

    Iteratively rejects downward excursions at ``mask_sigma`` robust sigmas
    until the retained population is no longer transit-dominated. A transit is a
    small fraction of the samples but sits many sigma below the baseline, so a
    single pass is enough in practice; the loop bounds pathological cases.
    """
    resid = detrended.copy()
    for _ in range(5):
        med = np.median(resid)
        mad = np.median(np.abs(resid - med))
        sigma = 1.4826 * mad
        if sigma <= 0:
            break
        keep = resid > (med - mask_sigma * sigma)
        # Stop if masking starts eating the bulk of the series.
        if keep.mean() < 0.5 or keep.all():
            break
        if keep.sum() == resid.size:
            break
        resid = resid[keep]
    med = np.median(resid)
    mad = np.median(np.abs(resid - med))
    return resid, float(1.4826 * mad)


def empirical_null(
    lc: LightCurve,
    observed_snr: float,
    n_trials: int = 20,
    min_period_days: float = 0.1,
    max_period_days: float = 50.0,
    duration_grid: Optional[Sequence[float]] = None,
    detrend_window: int | None = None,
    outlier_sigma: float = 5.0,
    n_periods_science: int = 10000,
    n_periods_null: int = 2000,
    seed: int = 20261012,
    method: Literal["signal-masked", "phase-shuffle"] = "signal-masked",
) -> NullDistribution:
    """Build an empirical null for the BLS selection statistic.

    Two surrogates are available and they are *not* interchangeable:

    ``signal-masked`` (default)
        Residuals are detrended, the transit is masked off at a robust
        threshold, and the surviving values are shuffled. This removes the
        signal while keeping the real noise distribution, so it answers the
        question we actually care about: how large a box could this noise
        support by chance.

    ``phase-shuffle``
        All detrended values are shuffled, including the transit's. This
        preserves every value but scatters the dip's samples across the
        series, where BLS can still find box-like structure among randomly
        placed extreme points. The resulting null is *conservative* — it cannot
        manufacture a false positive — but it can mask a real detection.

    Measured on the demo light curve (real BLS SNR 514): the signal-masked null
    peaks near SNR 5 while the phase-shuffle null peaks near 34, because 2.7% of
    samples sit below -4000 ppm purely from being the transit. With the
    default 20 trials the p-value floor is 1/21 = 0.048, so neither surrogate can
    express a p-value below that; it is reported with every result.
    """
    time, flux, flux_err = _clean_lightcurve(lc, outlier_sigma=outlier_sigma)

    from .features import _median_cadence_days, required_detrend_window_points

    cadence = _median_cadence_days(time)
    grid = list(duration_grid) if duration_grid is not None else [0.05, 0.1, 0.15, 0.2]
    if detrend_window is None:
        detrend_window = required_detrend_window_points(cadence, max(grid))

    detrended = _detrend(time, flux, window=detrend_window)
    detrended = detrended - np.median(detrended)

    if method == "signal-masked":
        pool, _ = _signal_free_residuals(detrended)
        if pool.size < max(200, 0.3 * detrended.size):
            # Too few signal-free samples to trust; fall back and say so.
            logger.warning(
                "Only %d/%d samples survive transit masking; using phase-shuffle null",
                pool.size, detrended.size,
            )
            pool = detrended
            method = "phase-shuffle"
    else:
        pool = detrended

    sci_periods = np.linspace(min_period_days, max_period_days, n_periods_science)
    null_periods = np.linspace(min_period_days, max_period_days, n_periods_null)
    durations = np.asarray(grid)

    rng = np.random.default_rng(seed)
    null_snr = np.empty(n_trials, dtype=float)
    n_usable = 0
    for _ in range(n_trials):
        shuffled = rng.permutation(pool)
        if shuffled.size != time.size:
            # Match lengths so the time base stays valid.
            if shuffled.size < time.size:
                shuffled = np.resize(shuffled, time.size)
            else:
                shuffled = shuffled[: time.size]
        s = _bls_peak_snr(time, shuffled, flux_err, null_periods, durations)
        if np.isfinite(s):
            null_snr[n_usable] = s
            n_usable += 1
    null_snr = null_snr[:n_usable]

    if n_usable == 0:
        return NullDistribution(
            observed_snr=observed_snr, n_trials=0, null_snr=np.array([]),
            p_value=1.0, p_value_floor=1.0, n_exceeding=0,
            grid_ratio=n_periods_null / n_periods_science, seed=seed,
            method=method,
            note="no usable null trials; p-value not resolvable",
        )

    n_exceeding = int(np.sum(null_snr >= observed_snr))
    p_value = (1 + n_exceeding) / (1 + n_usable)
    grid_ratio = n_periods_null / n_periods_science

    note = (
        f"p-value floor {1/(1+n_usable):.4f} at {n_usable} trials; "
        f"null period grid is {grid_ratio:.2f}x coarser than the science run, "
        "so the extreme tail is mildly optimistic"
    )
    if method == "phase-shuffle":
        note += "; phase-shuffle retains the transit values and is conservative"

    return NullDistribution(
        observed_snr=observed_snr,
        n_trials=n_usable,
        null_snr=null_snr,
        p_value=float(p_value),
        p_value_floor=float(1 / (1 + n_usable)),
        n_exceeding=n_exceeding,
        grid_ratio=grid_ratio,
        seed=seed,
        method=f"{method} BLS selection",
        note=note,
    )


# ── Independent time-segment holdout ──────────────────────────────────────


@dataclass
class SegmentHoldout:
    """Detect on one half, look for the signal in the untouched other half."""

    detected: bool
    segment_a_period_days: float
    segment_a_epoch: float
    segment_a_snr: float
    segment_b_snr: float
    segment_b_depth_ppm: float
    period_consistency_pct: float
    n_transits_expected: int
    n_transits_observed: int
    agrees: bool
    verdict: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "detected": self.detected,
            "segment_a_period_days": self.segment_a_period_days,
            "segment_a_snr": self.segment_a_snr,
            "segment_b_snr": self.segment_b_snr,
            "segment_b_depth_ppm": self.segment_b_depth_ppm,
            "period_consistency_pct": self.period_consistency_pct,
            "n_transits_expected": self.n_transits_expected,
            "n_transits_observed": self.n_transits_observed,
            "agrees": self.agrees,
            "verdict": self.verdict,
        }


def _fit_epoch(
    time: np.ndarray,
    flux: np.ndarray,
    period_days: float,
    duration_days: float,
    detrend_window: int = 101,
    epoch_hint: float = 0.0,
    n_offsets: int = 200,
) -> float:
    """Find the epoch that maximises depth in one light curve segment.

    Uses a phase scan rather than a full BLS so it is cheap and, more
    importantly, cannot return a different period — only the timing.
    """
    if len(time) < 10 or period_days <= 0:
        return float("nan")
    offsets = np.linspace(0.0, period_days, n_offsets, endpoint=False)
    best_offset, best_depth = epoch_hint % period_days, -np.inf
    for off in offsets:
        phase = ((time - off) % period_days) / period_days
        inner = phase < (duration_days / period_days)
        outer = (phase > 0.25) & (phase < 0.75)
        if inner.sum() < 2 or outer.sum() < 10:
            continue
        depth = np.median(flux[outer]) - np.median(flux[inner])
        if depth > best_depth:
            best_depth, best_offset = depth, off
    return float(best_offset)


def segment_holdout(
    lc: LightCurve,
    period_days: float,
    epoch: float,
    duration_days: float,
    split: float = 0.5,
    detrend_window: int | None = None,
    outlier_sigma: float = 5.0,
    min_snr_in_segment_b: float = 3.0,
) -> SegmentHoldout:
    """Fit on the first half of the baseline, test on the second half.

    The blind half is never used to choose the period or the epoch, so a
    coincidence in the fitting half cannot manufacture agreement.
    """
    from .features import _median_cadence_days, required_detrend_window_points

    time, flux, flux_err = _clean_lightcurve(lc, outlier_sigma=outlier_sigma)
    order = np.argsort(time)
    time, flux, flux_err = time[order], flux[order], flux_err[order]
    mid = int(len(time) * split)

    if detrend_window is None:
        detrend_window = required_detrend_window_points(
            _median_cadence_days(time), duration_days
        )
    order = np.argsort(time)
    time, flux, flux_err = time[order], flux[order], flux_err[order]
    mid = int(len(time) * split)

    time_a, flux_a, err_a = time[:mid], flux[:mid], flux_err[:mid]
    time_b, flux_b, err_b = time[mid:], flux[mid:], flux_err[mid:]

    detr_a = _detrend(time_a, flux_a, window=detrend_window)
    detr_a = detr_a - np.median(detr_a)
    detr_b = _detrend(time_b, flux_b, window=detrend_window)
    detr_b = detr_b - np.median(detr_b)

    # Fit the epoch on segment A only. The blind half is never consulted for
    # the ephemeris, so agreement in segment B is genuine out-of-sample evidence
    # rather than a re-fit.
    epoch_fit = _fit_epoch(
        time_a, detr_a, period_days, duration_days,
        detrend_window=detrend_window,
        epoch_hint=epoch,
    )
    epoch_used = epoch_fit if np.isfinite(epoch_fit) else epoch

    # Measure the signal at the fitted period and epoch in the blind half.
    phase = ((time_b - epoch_used) % period_days) / period_days
    in_transit = phase < (duration_days / period_days)
    n_expected = int((time_b[-1] - time_b[0]) / period_days) if len(time_b) > 1 else 0

    if in_transit.sum() < 3:
        return SegmentHoldout(
            detected=False, segment_a_period_days=period_days, segment_a_epoch=epoch,
            segment_a_snr=0.0, segment_b_snr=0.0, segment_b_depth_ppm=0.0,
            period_consistency_pct=0.0, n_transits_expected=n_expected,
            n_transits_observed=0, agrees=False,
            verdict="blind half too short for a transit",
        )

    baseline = detr_b[~in_transit]
    depth = float(np.median(baseline) - np.median(detr_b[in_transit]))
    rms = float(np.std(baseline))
    n_obs = int(in_transit.sum())
    snr_b = depth / (rms / np.sqrt(n_obs)) if rms > 0 else 0.0

    # Count distinct transits actually present in the blind half.
    transit_number = np.floor((time_b - epoch) / period_days).astype(int)
    n_observed = int(len(np.unique(transit_number[in_transit])))

    agrees = snr_b >= min_snr_in_segment_b
    if agrees:
        verdict = f"signal reappears in blind half (SNR {snr_b:.1f}, {n_observed} transit(s))"
    elif abs(depth) > 0 and n_obs >= 3:
        verdict = f"transit-like dip present but not significant in blind half (SNR {snr_b:.1f})"
    else:
        verdict = f"no signal in blind half (SNR {snr_b:.1f})"

    return SegmentHoldout(
        detected=True,
        segment_a_period_days=period_days,
        segment_a_epoch=epoch_used,
        segment_a_snr=0.0,
        segment_b_snr=float(snr_b),
        segment_b_depth_ppm=float(depth * 1e6),
        period_consistency_pct=0.0,
        n_transits_expected=n_expected,
        n_transits_observed=n_observed,
        agrees=agrees,
        verdict=verdict,
    )


# ── Injected-signal recovery benchmark ────────────────────────────────────


@dataclass
class InjectionResult:
    """Outcome for one injected signal."""

    snr_injected: float
    period_injected: float
    depth_injected_ppm: float
    duration_hours: float
    detected: bool
    period_recovered: float
    depth_recovered_ppm: float
    period_error_pct: float
    depth_error_pct: float
    recovered_snr: float
    n_runs: int
    n_detected: int

    @property
    def detection_rate(self) -> float:
        return self.n_detected / self.n_runs if self.n_runs else 0.0


@dataclass
class InjectionBenchmark:
    """Positive control: can the pipeline recover signals we put in?"""

    results: list[InjectionResult]
    detect_threshold_snr: float

    @property
    def detection_rate(self) -> float:
        """Rate at or above the pipeline's own detection threshold."""
        return self.detection_rate_at(self.detect_threshold_snr)

    def detection_rate_at(self, snr: float) -> float:
        eligible = [r for r in self.results if r.snr_injected >= snr]
        if not eligible:
            return 0.0
        total = sum(r.n_runs for r in eligible)
        hits = sum(r.n_detected for r in eligible)
        return hits / total if total else 0.0

    @property
    def median_period_error_pct(self) -> float:
        detected = [r for r in self.results if r.detected]
        if not detected:
            return float("nan")
        return float(np.median([r.period_error_pct for r in detected]))

    def sensitivity_limit_snr(self, target_rate: float = 0.8) -> Optional[float]:
        """Lowest injected SNR whose own detection rate reaches the target.

        Uses each point's rate directly, not the cumulative rate, so a weak SNR
        that failed cannot be masked by stronger ones clearing the bar.
        """
        for r in sorted(self.results, key=lambda x: x.snr_injected):
            if r.detection_rate >= target_rate:
                return r.snr_injected
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [vars(r) for r in self.results],
            "detection_rate": self.detection_rate,
            "median_period_error_pct": self.median_period_error_pct,
        }


def _make_flat_lc(
    source_id: str,
    n_points: int,
    noise_ppm: float,
    seed: int,
    cadence_minutes: float = 2.0,
) -> LightCurve:
    """Flat Gaussian-noise light curve, same cadence convention as the demo source."""
    from datetime import UTC, datetime

    from ..models import DataSource

    rng = np.random.default_rng(seed)
    time = np.arange(n_points) * (cadence_minutes / 1440.0)
    flux = np.ones(n_points) + rng.normal(0, noise_ppm * 1e-6, n_points)
    flux_err = np.full(n_points, noise_ppm * 1e-6)
    return LightCurve(
        source_id=source_id, time=time, flux=flux, flux_err=flux_err,
        source=DataSource.DEMO, cadence_seconds=cadence_minutes * 60,
        retrieved_at=datetime.now(UTC),
    )


def inject_transit(
    lc: LightCurve,
    period_days: float,
    depth_ppm: float,
    duration_hours: float,
    epoch_offset: float = 0.05,
) -> LightCurve:
    """Add a box-shaped transit to a copy of the light curve."""
    time = lc.time.copy()
    flux = lc.flux.copy()
    depth = depth_ppm * 1e-6
    duration = duration_hours / 24.0
    phase = ((time - epoch_offset) % period_days) / period_days
    flux[phase < (duration / period_days)] -= depth
    out = LightCurve(
        source_id=lc.source_id, time=time, flux=flux, flux_err=lc.flux_err.copy(),
        source=lc.source, cadence_seconds=lc.cadence_seconds, metadata=dict(lc.metadata),
    )
    return out


def injection_benchmark(
    snr_grid: Sequence[float] = (3.0, 5.0, 7.0, 9.0, 12.0, 15.0),
    period_days: float = 3.7,
    duration_hours: float = 2.5,
    noise_ppm: float = 200.0,
    cadence_minutes: float = 10.0,
    n_transits: int = 4,
    min_period_days: float = 0.5,
    max_period_days: float = 15.0,
    detect_threshold_snr: float = 7.0,
    n_seeds: int = 3,
    seed: int = 20261012,
    n_periods: int = 3000,
) -> InjectionBenchmark:
    """Measure recovery of signals whose parameters we chose.

    The baseline is sized from ``n_transits`` so every injected signal is fully
    sampled; a shorter window would let the benchmark measure its own
    insufficient baseline rather than the pipeline's sensitivity.

    ``snr_grid`` is converted to a depth using ``noise_ppm`` and the number of
    in-transit cadences, so the grid is a true SNR grid rather than an
    arbitrary list of depths.
    """
    from .features import (
        _clean_lightcurve,
        _detrend,
        _median_cadence_days,
        required_detrend_window_points,
    )

    cadence_days = cadence_minutes / 1440.0
    n_points = int(np.ceil(n_transits * period_days / cadence_days))
    n_in_transit = max(int(duration_hours / 24.0 / cadence_days), 1)
    per_cadence_err = noise_ppm * 1e-6
    sigma = per_cadence_err / np.sqrt(n_in_transit)

    # Detrend window must exceed the transit or the trend absorbs it.
    detrend_win = required_detrend_window_points(cadence_days, duration_hours / 24.0)

    results: list[InjectionResult] = []
    for snr_target in snr_grid:
        depth = snr_target * sigma
        depth_ppm = depth * 1e6
        detected_count = 0
        period_errs: list[float] = []
        depth_errs: list[float] = []
        last_period = float("nan")
        last_depth = float("nan")
        last_snr = float("nan")

        for s in range(n_seeds):
            flat = _make_flat_lc(
                f"inject_s{snr_target}_{s}", n_points, noise_ppm, seed + s,
                cadence_minutes=cadence_minutes,
            )
            injected = inject_transit(flat, period_days, depth_ppm, duration_hours)

            t, f, fe = _clean_lightcurve(injected)
            d = _detrend(t, f, window=detrend_win)
            d = d - np.median(d)

            periods = np.linspace(min_period_days, max_period_days, n_periods)
            durations = np.array([0.05, 0.1, 0.15, 0.2])
            durations = durations[durations < period_days]
            peak_snr = _bls_peak_snr(t, d, fe, periods, durations)

            if peak_snr >= detect_threshold_snr:
                from astropy.timeseries import BoxLeastSquares

                bls = BoxLeastSquares(t, d, dy=fe)
                power = bls.power(periods, durations)
                idx = int(np.argmax(power.power))
                last_period = float(power.period[idx])
                stats = bls.compute_stats(
                    last_period, float(power.duration[idx]), float(power.transit_time[idx])
                )
                dt = stats["depth"]
                if isinstance(dt, (tuple, list)) and len(dt) >= 2:
                    last_depth = float(dt[0]) * 1e6
                    last_snr = float(dt[0]) / float(dt[1]) if dt[1] > 0 else 0.0
                detected_count += 1
                if last_period > 0:
                    period_errs.append(abs(last_period - period_days) / period_days * 100)
                if depth > 0:
                    depth_errs.append(abs(last_depth - depth_ppm) / depth_ppm * 100)

        results.append(
            InjectionResult(
                snr_injected=snr_target,
                period_injected=period_days,
                depth_injected_ppm=depth_ppm,
                duration_hours=duration_hours,
                detected=detected_count > 0,
                period_recovered=last_period,
                depth_recovered_ppm=last_depth,
                period_error_pct=float(np.median(period_errs)) if period_errs else float("nan"),
                depth_error_pct=float(np.median(depth_errs)) if depth_errs else float("nan"),
                recovered_snr=last_snr,
                n_runs=n_seeds,
                n_detected=detected_count,
            )
        )

    return InjectionBenchmark(
        results=results, detect_threshold_snr=detect_threshold_snr
    )


# ── Preprocessing sensitivity sweep ───────────────────────────────────────


@dataclass
class SensitivityResult:
    """Does the detection survive changing the preprocessing?"""

    n_variants: int
    n_surviving: int
    survival_rate: float
    period_spread_pct: float
    snr_range: tuple[float, float]
    variants: list[dict[str, Any]] = field(default_factory=list)
    verdict: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_variants": self.n_variants,
            "n_surviving": self.n_surviving,
            "survival_rate": self.survival_rate,
            "period_spread_pct": self.period_spread_pct,
            "snr_range": list(self.snr_range),
            "verdict": self.verdict,
            "variants": self.variants,
        }


def sensitivity_sweep(
    lc: LightCurve,
    min_period_days: float = 0.5,
    max_period_days: float = 20.0,
    detrend_multipliers: Sequence[float] = (1.0, 1.5, 2.0),
    outlier_sigmas: Sequence[float] = (3.0, 5.0, 8.0),
    n_periods: int = 3000,
) -> SensitivityResult:
    """Re-run BLS across preprocessing settings and report how stable the peak is.

    The detrend windows are treated as *multipliers of the minimum safe
    window* for the light curve's own cadence, so the sweep probes genuine
    sensitivity to preprocessing rather than re-testing the known
    trend-absorbs-transit artefact.
    """
    from .features import (
        _clean_lightcurve,
        _detrend,
        _median_cadence_days,
        required_detrend_window_points,
    )

    periods = np.linspace(min_period_days, max_period_days, n_periods)
    durations = np.array([0.05, 0.1, 0.15, 0.2])
    keep = durations < min_period_days
    if np.any(keep):
        durations = durations[keep]

    t_probe, _, _ = _clean_lightcurve(lc)
    cadence = _median_cadence_days(t_probe)
    safe_window = required_detrend_window_points(cadence, float(durations.max()))

    variants: list[dict[str, Any]] = []
    for win_mult in detrend_multipliers:
        win = max(int(round(safe_window * win_mult)), 11)
        if win % 2 == 0:
            win += 1
        for sigma in outlier_sigmas:
            t, f, fe = _clean_lightcurve(lc, outlier_sigma=sigma)
            d = _detrend(t, f, window=win)
            d = d - np.median(d)
            snr = _bls_peak_snr(t, d, fe, periods, durations)

            from astropy.timeseries import BoxLeastSquares

            period = float("nan")
            depth_ppm = float("nan")
            if np.isfinite(snr):
                bls = BoxLeastSquares(t, d, dy=fe)
                power = bls.power(periods, durations)
                idx = int(np.argmax(power.power))
                period = float(power.period[idx])
                st = bls.compute_stats(period, float(power.duration[idx]), float(power.transit_time[idx]))
                dt = st["depth"]
                if isinstance(dt, (tuple, list)) and len(dt) >= 2:
                    depth_ppm = float(dt[0]) * 1e6

            variants.append(
                {
                    "detrend_window": win,
                    "outlier_sigma": sigma,
                    "peak_snr": float(snr) if np.isfinite(snr) else float("nan"),
                    "period_days": period,
                    "depth_ppm": depth_ppm,
                }
            )

    finite = [v for v in variants if np.isfinite(v["peak_snr"]) and v["peak_snr"] > 0]
    snrs = [v["peak_snr"] for v in finite]
    periods_found = [v["period_days"] for v in finite if np.isfinite(v["period_days"])]

    # Survival threshold: the same 7.0 SNR used by the benchmark.
    surviving = [v for v in finite if v["peak_snr"] >= 7.0]
    spread = (
        float((max(periods_found) - min(periods_found)) / np.median(periods_found) * 100)
        if len(periods_found) > 1
        else 0.0
    )

    survival_rate = len(surviving) / len(variants) if variants else 0.0
    if survival_rate >= 0.8 and spread < 5.0:
        verdict = f"stable: survives {len(surviving)}/{len(variants)} settings, period spread {spread:.1f}%"
    elif survival_rate >= 0.5:
        verdict = f"marginal: survives {len(surviving)}/{len(variants)} settings, period spread {spread:.1f}%"
    else:
        verdict = f"unstable: survives only {len(surviving)}/{len(variants)} settings"

    return SensitivityResult(
        n_variants=len(variants),
        n_surviving=len(surviving),
        survival_rate=survival_rate,
        period_spread_pct=spread,
        snr_range=(min(snrs), max(snrs)) if snrs else (float("nan"), float("nan")),
        variants=variants,
        verdict=verdict,
    )


# ── Orchestration ─────────────────────────────────────────────────────────


@dataclass
class ValidationReport:
    """Everything known about one candidate's robustness."""

    candidate_id: str
    source_id: str
    null_result: Optional[NullDistribution]
    holdout: Optional[SegmentHoldout]
    sensitivity: Optional[SensitivityResult]
    fdr: Optional[FDRResult]
    verdict: str
    reasons: list[str] = field(default_factory=list)

    @property
    def robust(self) -> bool:
        return self.verdict == "robust"

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "source_id": self.source_id,
            "null_result": self.null_result.to_dict() if self.null_result else None,
            "holdout": self.holdout.to_dict() if self.holdout else None,
            "sensitivity": self.sensitivity.to_dict() if self.sensitivity else None,
            "fdr": vars(self.fdr) if self.fdr else None,
            "verdict": self.verdict,
            "reasons": self.reasons,
        }


def validate_candidate(
    candidate: Any,
    lc: LightCurve,
    n_null_trials: int = 20,
    alpha: float = 0.05,
    fdr_method: Literal["BY", "BH"] = "BY",
    min_period_days: float = 0.5,
    max_period_days: float = 20.0,
    duration_grid: Optional[Sequence[float]] = None,
    run_holdout: bool = True,
    run_sensitivity: bool = False,
    seed: int = 20261012,
) -> ValidationReport:
    """Run the per-candidate checks. FDR is applied separately across a family."""
    reasons: list[str] = []

    null_result = empirical_null(
        lc,
        observed_snr=candidate.snr,
        n_trials=n_null_trials,
        min_period_days=min_period_days,
        max_period_days=max_period_days,
        duration_grid=duration_grid,
        seed=seed,
    )
    if null_result.p_value >= 0.05:
        reasons.append(
            f"empirical null p={null_result.p_value:.3f} (floor {null_result.p_value_floor:.3f}) not significant"
        )

    holdout = None
    if run_holdout and candidate.duration_hours:
        holdout = segment_holdout(
            lc,
            period_days=candidate.period_days,
            epoch=candidate.epoch_bjd,
            duration_days=candidate.duration_hours / 24.0,
            detrend_window=101,
        )
        if not holdout.agrees:
            reasons.append(f"holdout: {holdout.verdict}")

    sensitivity = None
    if run_sensitivity:
        sensitivity = sensitivity_sweep(
            lc,
            min_period_days=min_period_days,
            max_period_days=max_period_days,
        )
        if sensitivity.survival_rate < 0.5:
            reasons.append(f"sensitivity: {sensitivity.verdict}")

    verdict = "robust" if not reasons else "weak"
    return ValidationReport(
        candidate_id=candidate.candidate_id,
        source_id=candidate.source_id,
        null_result=null_result,
        holdout=holdout,
        sensitivity=sensitivity,
        fdr=None,
        verdict=verdict,
        reasons=reasons,
    )


def apply_family_fdr(
    reports: Sequence[ValidationReport],
    alpha: float = 0.05,
    method: Literal["BY", "BH"] = "BY",
) -> list[ValidationReport]:
    """Apply one FDR correction across the whole candidate family.

    Reports without a resolvable p-value are excluded from the family and are
    reported as such rather than being silently counted as non-significant.
    """
    usable = [r for r in reports if r.null_result is not None and r.null_result.n_trials > 0]
    if not usable:
        for r in reports:
            r.fdr = None
        return list(reports)

    p_values = [r.null_result.p_value for r in usable]
    labels = [r.candidate_id for r in usable]
    fdr = (
        benjamini_yekutieli(p_values, labels, alpha)
        if method == "BY"
        else benjamini_hochberg(p_values, labels, alpha)
    )

    rejected = set(fdr.rejected)
    for r in usable:
        r.fdr = fdr
        if r.candidate_id not in rejected and not r.reasons:
            r.verdict = "weak"
            r.reasons.append(f"not rejected by {method} at alpha={alpha} (family of {fdr.n_tests})")
    return list(reports)