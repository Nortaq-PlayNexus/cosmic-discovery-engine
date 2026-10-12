"""Characterise the two detrend/clip defects and the robustness of the fixes.

Run:  python scripts/characterize_defects.py --quick

Design notes
------------
* Depth recovery is measured at the KNOWN transit phase, not as ``np.min``.
  An earlier diagnostic used np.min over the whole series, which is dominated
  by noise and produced a false exoneration of the detrend window.
* Recovery is measured without running BLS wherever possible, because BLS is
  the expensive part and the question here is a preprocessing question.
* Every sweep has an explicit control arm (ratio far above the transit, and
  flat noise) so a "recovery" number can be checked against something.
* Compute is bounded: max points, max periods, fixed grid sizes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cosmic_discovery.pipeline.features import (  # noqa: E402
    _clean_lightcurve,
    _detrend,
    _median_cadence_days,
    required_detrend_window_points,
)
from cosmic_discovery.pipeline.synthetic import (  # noqa: E402
    flat_series,
    inject_box,
)

# Bounded-compute limits (machine has 23.9 GB; never allocate unbounded grids).
MAX_POINTS = 4000
MAX_BLS_PERIODS = 2000


def measure_depth(time, flux, period, epoch, duration_days):
    """Depth at the known ephemeris, in ppm. Returns NaN if unusable."""
    phase = ((time - epoch) % period) / period
    inner = phase < (duration_days / period)
    outer = (phase > 0.25) & (phase < 0.75)
    if inner.sum() < 3 or outer.sum() < 20:
        return float("nan")
    return float((np.median(flux[outer]) - np.median(flux[inner])) * 1e6)


def question1_window_sweep(quick=False):
    """Q1: recovery vs detrend-window / transit-duration ratio.

    Falsification test: if the defect were not real, recovery would be flat in
    the ratio. If a too-long window creates its own artifact, recovery would
    fall again at the top of the range. Both are checked.
    """
    ratios = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0]
    if quick:
        ratios = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 16.0]
    durations_h = [1.5, 3.0, 6.0, 12.0]
    depths_ppm = [300.0, 1000.0, 5000.0]
    cadence_min = 10.0
    period_days = 3.7
    cadence_days = cadence_min / 1440.0
    n_points = min(MAX_POINTS, int(6 * period_days / cadence_days))
    noise_ppm = 200.0

    print("=" * 78)
    print("Q1  Recovery (%) vs detrend-window / transit-duration ratio")
    print("    100% = injected depth recovered.  Control: ratio>>1 should be ~100.")
    print("=" * 78)
    header = f"{'ratio':>6} |" + "".join(
        f" {f'{d:g}h/{dp:.0f}ppm':>13}" for d in durations_h for dp in depths_ppm[:2]
    )
    print(header)
    print("-" * len(header))

    results = {}
    for ratio in ratios:
        cells = []
        for dur_h in durations_h:
            dur_days = dur_h / 24.0
            window = int(round(ratio * dur_days / cadence_days))
            window = max(window + 1 if window % 2 == 0 else window, 3)
            for depth_ppm in depths_ppm[:2]:
                t, f = flat_series(n_points, cadence_min, noise_ppm, seed=1)
                ti, fi = inject_box(t, f, period_days, depth_ppm, dur_days)
                d = _detrend(ti, fi, window=window)
                d = d - np.median(d)
                got = measure_depth(ti, d, period_days, 0.05, dur_days)
                rec = 100.0 * got / depth_ppm if depth_ppm > 0 and np.isfinite(got) else float("nan")
                cells.append(rec)
                results[(ratio, dur_h, depth_ppm)] = rec
        print(f"{ratio:>6.2f} |" + "".join(f" {c:>12.1f}%" for c in cells))

    # Where does recovery cross 90%?
    print()
    for dur_h in durations_h:
        for depth_ppm in depths_ppm[:2]:
            key = next(((r, results[(r, dur_h, depth_ppm)]) for r in ratios
                        if np.isfinite(results[(r, dur_h, depth_ppm)])
                        and results[(r, dur_h, depth_ppm)] >= 90.0), None)
            tag = f"ratio >= {key[0]:.2f}" if key else "never reaches 90%"
            print(f"  {dur_h:4.1f} h / {depth_ppm:5.0f} ppm: 90% recovery at {tag}")
    print()
    return results


def question2_cadence_depth(quick=False):
    """Q2: is the defect cadence- and depth-dependent, i.e. is the shipped
    default really safe at 30 min cadence and only breaks at 2 min?

    Also checks whether a long window starts costing recovery (its own artifact).
    """
    cadences = [2.0, 10.0, 30.0]
    if quick:
        cadences = [2.0, 30.0]
    dur_h = 6.0
    depth_ppm = 1000.0
    period_days = 3.7
    dur_days = dur_h / 24.0

    print("=" * 78)
    print("Q2  Effect of cadence on the shipped default (window=101 points)")
    print("=" * 78)
    print(f"{'cadence':>9} {'window days':>12} {'ratio to 6h transit':>19} {'recovery':>10}")
    print("-" * 56)
    for cad_min in cadences:
        cadence_days = cad_min / 1440.0
        n_points = min(MAX_POINTS, int(6 * period_days / cadence_days))
        t, f = flat_series(n_points, cad_min, 200.0, seed=2)
        ti, fi = inject_box(t, f, period_days, depth_ppm, dur_days)
        d = _detrend(ti, fi, window=101)
        d = d - np.median(d)
        got = measure_depth(ti, d, period_days, 0.05, dur_days)
        rec = 100.0 * got / depth_ppm
        print(f"{cad_min:>7.0f}m {101*cadence_days:>12.4f} "
              f"{101*cadence_days/dur_days:>19.2f} {rec:>9.1f}%")

    print()
    print("  Safe window the auto-correction would choose:")
    for cad_min in cadences:
        w = required_detrend_window_points(cad_min / 1440.0, dur_days)
        print(f"    {cad_min:>3.0f} min cadence -> {w} points "
              f"({w * cad_min/1440.0:.3f} d, {w*cad_min/1440.0/dur_days:.1f}x transit)")
    print()


def question3_cross_validation(quick=False):
    """Q3: cross-validate with an INDEPENDENT detrend method.

    If the finding is an artifact of scipy's median_filter it will not
    reproduce under a different estimator. Compares against a masked
    iterative median (the standard transit-safe approach) and a Savitzky-Golay.
    """
    from scipy.signal import savgol_filter

    dur_h = 6.0
    depth_ppm = 1000.0
    period_days = 3.7
    dur_days = dur_h / 24.0
    cadence_min = 10.0
    cadence_days = cadence_min / 1440.0
    n_points = min(MAX_POINTS, int(6 * period_days / cadence_days))

    t, f = flat_series(n_points, cadence_min, 200.0, seed=3)
    ti, fi = inject_box(t, f, period_days, depth_ppm, dur_days)

    def masked_median_detrend(t, f, window):
        """Iteratively mask points below the running trend, then detrend."""
        keep = np.ones(len(f), dtype=bool)
        trend = None
        for _ in range(3):
            seg = f[keep]
            trend_full = np.empty_like(f)
            # interpolate the kept values onto the full grid, then median filter
            idx = np.arange(len(f))
            filled = np.interp(idx, idx[keep], seg)
            trend_full = median_filter_pad(filled, window)
            resid = f - trend_full
            med, mad = np.median(resid), np.median(np.abs(resid - np.median(resid)))
            sigma = 1.4826 * mad
            if sigma <= 0:
                break
            new_keep = resid > (med - 5 * sigma)
            if new_keep.mean() < 0.5 or (new_keep == keep).all():
                break
            keep = new_keep
        return f / trend_full - 1.0

    def median_filter_pad(x, w):
        from scipy.ndimage import median_filter
        w = max(int(w), 3)
        w = w + 1 if w % 2 == 0 else w
        return median_filter(x, size=w, mode="reflect")

    print("=" * 78)
    print("Q3  Cross-validation: does the finding reproduce under other detrenders?")
    print("=" * 78)
    print(f"  injected: {depth_ppm:.0f} ppm, {dur_h} h, P={period_days} d")
    print()

    short_w = int(round(1.0 * dur_days / cadence_days))
    long_w = int(round(6.0 * dur_days / cadence_days))
    long_w = long_w + 1 if long_w % 2 == 0 else long_w

    # scipy median, short window (the defect regime)
    d = _detrend(ti, fi, window=max(short_w, 3))
    d = d - np.median(d)
    r1 = 100 * measure_depth(ti, d, period_days, 0.05, dur_days) / depth_ppm

    # scipy median, long window (the fix)
    d = _detrend(ti, fi, window=long_w)
    d = d - np.median(d)
    r2 = 100 * measure_depth(ti, d, period_days, 0.05, dur_days) / depth_ppm

    # masked iterative median, short window
    d = masked_median_detrend(ti, fi, window=max(short_w, 3))
    r3 = 100 * measure_depth(ti, d, period_days, 0.05, dur_days) / depth_ppm

    # savgol with a short window
    wl = max(int(short_w) | 1, 5)
    if wl >= 5 and wl < len(fi):
        sm = savgol_filter(fi, wl, polyorder=2, mode="interp")
        d = fi / sm - 1.0
        r4 = 100 * measure_depth(ti, d, period_days, 0.05, dur_days) / depth_ppm
    else:
        r4 = float("nan")

    print(f"  {'scipy median, short window':<34} {r1:>6.1f}%")
    print(f"  {'scipy median, long window':<34} {r2:>6.1f}%")
    print(f"  {'masked iterative median, short':<34} {r3:>6.1f}%")
    print(f"  {'savgol, short window':<34} {r4:>6.1f}%")
    print()
    if r1 < 50 and r2 > 80:
        print("  -> finding REPRODUCES: short window loses the transit, long window keeps it.")
    else:
        print("  -> finding did NOT reproduce cleanly; treat with suspicion.")
    print()


def question4_downward_glitches(quick=False):
    """Q4 (adversarial): does one-sided clipping leave downward glitches that
    BLS mistakes for transits? This is the COST of the fix.

    A downward instrumental glitch is indistinguishable in shape from a
    shallow single transit. Quantify whether the no-downward-clip policy turns
    glitches into false positives.
    """
    from cosmic_discovery.pipeline.validation import _bls_peak_snr

    cadence_min = 10.0
    cadence_days = cadence_min / 1440.0
    n_points = min(MAX_POINTS, int(8 * 3.0 / cadence_days))
    periods = np.linspace(0.5, 15.0, min(MAX_BLS_PERIODS, 1500))
    durations = np.array([0.05, 0.1, 0.15])
    rng = np.random.default_rng(11)

    n_glitches = [0, 3, 10, 30]
    snrs = []
    for n_g in n_glitches:
        trials = []
        for rep in range(4):
            t, f = flat_series(n_points, cadence_min, 200.0, seed=100 + rep)
            # Scattered downward glitches: contiguous short dips at random times.
            for _ in range(n_g):
                start = rng.integers(0, n_points - 20)
                f[start:start + 20] -= rng.uniform(0.001, 0.005)
            from cosmic_discovery.models import DataSource, LightCurve
            lc = LightCurve(
                source_id=f"g{n_g}_{rep}", time=t, flux=f,
                flux_err=np.full(n_points, 200e-6), source=DataSource.DEMO,
            )
            tc, fc, ec = _clean_lightcurve(lc, outlier_sigma=5.0)
            d = _detrend(tc, fc, window=required_detrend_window_points(
                _median_cadence_days(tc), 0.15))
            d = d - np.median(d)
            trials.append(_bls_peak_snr(tc, d, ec, periods, durations))
        snrs.append(np.median(trials))

    print("=" * 78)
    print("Q4  Cost of one-sided clipping: scattered downward glitches")
    print("    (the pipeline no longer clips these; can they fake a transit?)")
    print("=" * 78)
    print(f"  {n_glitches and 'downward glitches':>20} | {'median BLS peak SNR':>18}")
    for n_g, s in zip(n_glitches, snrs):
        flag = "  <- above the SNR-7 detect threshold" if s >= 7 else ""
        print(f"  {n_g:>20} | {s:>18.1f}{flag}")
    print()
    print("  A rising SNR with glitch count would mean un-clipped dips accumulate")
    print("  into a spurious periodic signal.")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="coarser grids")
    args = ap.parse_args()

    q = [question1_window_sweep, question2_cadence_depth,
         question3_cross_validation, question4_downward_glitches]
    for fn in q:
        fn(quick=args.quick)


if __name__ == "__main__":
    main()