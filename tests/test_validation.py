"""Tests for the statistical validation layer.

The FDR procedures are checked against values worked out by hand, not against
the implementation itself, so a shared bug cannot make a test pass.
"""

from __future__ import annotations

import numpy as np
import pytest

from cosmic_discovery.database import Database
from cosmic_discovery.models import CandidateStatus, LightCurve, TransitCandidate, DataSource
from cosmic_discovery.pipeline.features import extract_features
from cosmic_discovery.pipeline.validation import (
    FDRResult,
    NullDistribution,
    apply_family_fdr,
    benjamini_hochberg,
    benjamini_yekutieli,
    empirical_null,
    inject_transit,
    injection_benchmark,
    segment_holdout,
    sensitivity_sweep,
    validate_candidate,
    _check_fdr_preconditions,
    _make_flat_lc,
)


# ── FDR preconditions ─────────────────────────────────────────────────────


def test_fdr_preconditions_reject_empty():
    ok, note = _check_fdr_preconditions([], 0.05)
    assert not ok
    assert "no tests" in note


def test_fdr_preconditions_reject_out_of_range():
    ok, note = _check_fdr_preconditions([0.1, 1.5], 0.05)
    assert not ok
    assert "outside [0, 1]" in note


def test_fdr_preconditions_reject_nonfinite():
    ok, note = _check_fdr_preconditions([0.1, float("nan")], 0.05)
    assert not ok
    assert "non-finite" in note


def test_fdr_preconditions_reject_bad_alpha():
    ok, note = _check_fdr_preconditions([0.1, 0.2], 0.0)
    assert not ok
    assert "alpha" in note


def test_fdr_preconditions_accept_valid():
    ok, note = _check_fdr_preconditions([0.01, 0.2, 0.5], 0.05)
    assert ok
    assert "satisfied" in note


def test_fdr_preconditions_flag_degenerate():
    ok, note = _check_fdr_preconditions([0.3, 0.3, 0.3], 0.05)
    assert ok
    assert "identical" in note


# ── Benjamini-Hochberg, hand-worked ───────────────────────────────────────


def test_bh_none_rejected():
    # All p-values large; nothing should pass p <= k/m * alpha
    # m=4, alpha=0.05 -> thresholds 0.0125, 0.025, 0.0375, 0.05
    # p = [0.2, 0.3, 0.4, 0.5] -> nothing passes
    result = benjamini_hochberg([0.2, 0.3, 0.4, 0.5], ["a", "b", "c", "d"], alpha=0.05)
    assert result.n_rejected == 0
    assert result.applicability_ok


def test_bh_all_rejected():
    # All p tiny -> all rejected
    result = benjamini_hochberg([0.0001, 0.0002, 0.0003], ["a", "b", "c"], alpha=0.05)
    assert result.n_rejected == 3
    assert set(result.rejected) == {"a", "b", "c"}


def test_bh_hand_worked_partial():
    # m=5, alpha=0.05
    # thresholds: 0.01, 0.02, 0.03, 0.04, 0.05
    # p sorted: 0.005, 0.015, 0.025, 0.6, 0.7
    #   p1=0.005 <= 0.01  pass
    #   p2=0.015 <= 0.02  pass
    #   p3=0.025 <= 0.03  pass
    #   p4=0.6   >  0.04  fail
    #   p5=0.7   >  0.05  fail
    # largest passing k = 3 -> reject the 3 smallest
    p = [0.005, 0.015, 0.025, 0.6, 0.7]
    labels = ["s1", "s2", "s3", "s4", "s5"]
    result = benjamini_hochberg(p, labels, alpha=0.05)
    assert result.n_rejected == 3
    assert set(result.rejected) == {"s1", "s2", "s3"}
    assert result.thresholds[:3] == pytest.approx([0.01, 0.02, 0.03])
    assert result.thresholds[3] == pytest.approx(0.04)


def test_bh_labels_track_input_order():
    # Input not sorted; labels must follow their p-value, not sort position
    p = [0.6, 0.005, 0.7, 0.015]
    labels = ["big1", "small1", "big2", "small2"]
    result = benjamini_hochberg(p, labels, alpha=0.05)
    # m=4 -> thresholds 0.0125, 0.025, 0.0375, 0.05
    # sorted p: 0.005(small1), 0.015(small2), 0.6, 0.7
    #   0.005 <= 0.0125 pass; 0.015 <= 0.025 pass; 0.6 > 0.0375; 0.7 > 0.05
    assert set(result.rejected) == {"small1", "small2"}


# ── Benjamini-Yekutieli, hand-worked ──────────────────────────────────────


def test_by_is_more_conservative_than_bh():
    p = [0.005, 0.015, 0.025, 0.03, 0.035]
    labels = [f"t{i}" for i in range(5)]
    bh = benjamini_hochberg(p, labels, alpha=0.05)
    by = benjamini_yekutieli(p, labels, alpha=0.05)
    assert by.n_rejected <= bh.n_rejected


def test_by_hand_worked():
    # m=3, alpha=0.05, C(m) = 1 + 1/2 + 1/3 = 1.8333
    # scaled thresholds: 0.05 * (1/3) / 1.8333 = 0.00909
    #                    0.05 * (2/3) / 1.8333 = 0.01818
    #                    0.05 * (1)   / 1.8333 = 0.02727
    # p = [0.004, 0.01, 0.5]
    #   0.004 <= 0.00909 pass
    #   0.01  <= 0.01818 pass
    #   0.5   >  0.02727 fail
    # largest k = 2
    result = benjamini_yekutieli([0.004, 0.01, 0.5], ["a", "b", "c"], alpha=0.05)
    assert result.method == "BY"
    assert result.n_rejected == 2
    assert set(result.rejected) == {"a", "b"}
    assert "harmonic" in result.applicability_note


def test_by_single_strong_signal():
    # m=10, one very strong signal
    p = [1e-8] + [0.5] * 9
    labels = ["strong"] + [f"w{i}" for i in range(9)]
    by = benjamini_yekutieli(p, labels, alpha=0.05)
    assert "strong" in by.rejected
    assert len(by.rejected) == 1


def test_by_rejects_nothing_when_all_weak():
    p = [0.3] * 5
    labels = [f"t{i}" for i in range(5)]
    by = benjamini_yekutieli(p, labels, alpha=0.05)
    # all identical p=0.3, largest threshold = 0.05/1.8286 = 0.0273 < 0.3
    assert by.n_rejected == 0


def test_by_rejects_nothing_when_applicability_fails():
    result = benjamini_yekutieli([0.1, 2.0], ["a", "b"], alpha=0.05)
    assert not result.applicability_ok
    assert result.n_rejected == 0
    assert result.rejected == []


# ── Empirical null ────────────────────────────────────────────────────────


def test_empirical_null_on_flat_noise_is_not_significant(flat_lc):
    """A flat light curve at modest SNR must not survive its own null."""
    null = empirical_null(flat_lc, observed_snr=5.0, n_trials=12, max_period_days=15.0)
    assert null.n_trials > 0
    assert null.p_value >= 0.05
    assert not null.significant_at


def test_empirical_null_on_transit_is_significant(demo_lc):
    """A real transit at its measured SNR must survive the null.

    Uses the SNR the pipeline actually reports rather than an invented number.
    With the default 20 trials the p-value floor is 1/21 = 0.0476, so 25 trials
    are used here to make p < 0.05 reachable.
    """
    features = extract_features(demo_lc, min_period_days=0.5, max_period_days=20.0)
    null = empirical_null(
        demo_lc, observed_snr=features.bls_snr, n_trials=25, max_period_days=20.0
    )
    assert null.p_value < 0.05
    assert null.significant_at
    assert null.n_exceeding == 0  # no surrogate came close


def test_signal_masked_null_is_tighter_than_phase_shuffle(demo_lc):
    """The signal-masked surrogate must sit below the phase-shuffle one.

    Phase shuffling keeps the transit's own values, which lets BLS find box
    structure among scattered extremes; masking removes them. Asserting the
    ordering pins that behaviour so a future change cannot silently swap them.
    """
    features = extract_features(demo_lc, min_period_days=0.5, max_period_days=20.0)
    masked = empirical_null(
        demo_lc, observed_snr=features.bls_snr, n_trials=12,
        max_period_days=20.0, method="signal-masked",
    )
    shuffled = empirical_null(
        demo_lc, observed_snr=features.bls_snr, n_trials=12,
        max_period_days=20.0, method="phase-shuffle",
    )
    assert "signal-masked" in masked.method
    assert "phase-shuffle" in shuffled.method
    assert masked.null_snr.max() < shuffled.null_snr.max()


def test_empirical_null_p_value_floor(demo_lc):
    """p-value can never go below 1/(1+n_trials)."""
    null = empirical_null(demo_lc, observed_snr=1e9, n_trials=8, max_period_days=15.0)
    assert null.p_value == pytest.approx(1 / (1 + null.n_trials))


def test_empirical_null_reports_floor(demo_lc):
    null = empirical_null(demo_lc, observed_snr=10.0, n_trials=10, max_period_days=15.0)
    assert null.p_value_floor == pytest.approx(1 / (1 + null.n_trials))
    assert "floor" in null.note


def test_empirical_null_is_reproducible(demo_lc):
    a = empirical_null(demo_lc, observed_snr=10.0, n_trials=10, max_period_days=15.0, seed=123)
    b = empirical_null(demo_lc, observed_snr=10.0, n_trials=10, max_period_days=15.0, seed=123)
    assert np.allclose(a.null_snr, b.null_snr)


def test_empirical_null_changes_with_seed(demo_lc):
    a = empirical_null(demo_lc, observed_snr=10.0, n_trials=10, max_period_days=15.0, seed=1)
    b = empirical_null(demo_lc, observed_snr=10.0, n_trials=10, max_period_days=15.0, seed=2)
    assert not np.allclose(a.null_snr, b.null_snr)


def test_empirical_null_reports_grid_ratio(demo_lc):
    null = empirical_null(
        demo_lc, observed_snr=10.0, n_trials=5, max_period_days=15.0,
        n_periods_science=10000, n_periods_null=2000,
    )
    assert null.grid_ratio == pytest.approx(0.2)
    assert "coarser" in null.note


# ── Segment holdout ───────────────────────────────────────────────────────


def test_segment_holdout_agrees_on_real_transit(demo_lc):
    """A genuine 5-day transit should reappear in the blind half."""
    hold = segment_holdout(demo_lc, period_days=5.0, epoch=0.05, duration_days=0.1)
    assert hold.detected
    assert hold.segment_b_snr > 3.0
    assert hold.agrees


def test_segment_holdout_rejects_wrong_period(flat_lc):
    """A period that does not exist must not reappear."""
    hold = segment_holdout(flat_lc, period_days=5.0, epoch=0.05, duration_days=0.1)
    assert not hold.agrees


def test_segment_holdout_short_baseline(demo_lc):
    """A period far longer than the baseline has too few transits to confirm."""
    hold = segment_holdout(demo_lc, period_days=500.0, epoch=0.05, duration_days=0.1)
    assert not hold.agrees
    assert "blind half" in hold.verdict


# ── Injection benchmark (positive control) ────────────────────────────────


def test_injection_benchmark_detects_strong_signals():
    """Positive control: strong injected signals must be recovered."""
    bench = injection_benchmark(
        snr_grid=(8.0, 12.0, 16.0), n_seeds=2, n_transits=4, n_periods=1500
    )
    assert len(bench.results) == 3
    for r in bench.results:
        if r.snr_injected >= 12.0:
            assert r.detection_rate == 1.0
            assert r.period_error_pct < 5.0


def test_injection_benchmark_detects_weak_signals_less_often():
    """Recovery rate must increase with injected SNR (monotone-ish)."""
    bench = injection_benchmark(
        snr_grid=(3.0, 6.0, 10.0, 14.0), n_seeds=3, n_transits=4, n_periods=1500
    )
    weak = bench.detection_rate_at(3.0)
    strong = bench.detection_rate_at(14.0)
    assert strong >= weak
    assert strong == 1.0


def test_injection_benchmark_recovers_period_accurately():
    bench = injection_benchmark(
        snr_grid=(14.0,), period_days=3.7, n_seeds=2, n_transits=4, n_periods=1500
    )
    r = bench.results[0]
    assert r.period_error_pct < 2.0
    assert r.depth_error_pct < 30.0


def test_injection_benchmark_has_sensitivity_limit():
    bench = injection_benchmark(
        snr_grid=(3.0, 5.0, 7.0, 10.0), n_seeds=2, n_transits=4, n_periods=1500
    )
    limit = bench.sensitivity_limit_snr(target_rate=0.8)
    assert limit is not None
    assert limit >= 3.0


def test_injection_transit_modifies_flux():
    flat = _make_flat_lc("t", 2000, 200.0, seed=1)
    injected = inject_transit(flat, period_days=2.0, depth_ppm=5000.0, duration_hours=2.0)
    assert not np.array_equal(flat.flux, injected.flux)
    assert injected.flux.min() < flat.flux.min()
    # original must be untouched
    assert np.array_equal(flat.flux, _make_flat_lc("t", 2000, 200.0, seed=1).flux)


def test_flat_lc_has_no_transit():
    flat = _make_flat_lc("t", 3000, 200.0, seed=7)
    med = np.median(flat.flux)
    assert np.min(flat.flux) > med * 0.99  # no deep dip


# ── Sensitivity sweep ─────────────────────────────────────────────────────


def test_sensitivity_sweep_on_transit(demo_lc):
    """A real transit should survive most preprocessing settings."""
    result = sensitivity_sweep(demo_lc, min_period_days=0.5, max_period_days=15.0, n_periods=1200)
    assert result.n_variants == 9
    assert result.survival_rate >= 0.5
    assert result.period_spread_pct < 25.0


def test_sensitivity_sweep_on_flat(flat_lc):
    """Noise should not look stable across preprocessing."""
    result = sensitivity_sweep(flat_lc, min_period_days=0.5, max_period_days=15.0, n_periods=1200)
    assert result.n_variants == 9
    assert result.verdict


# ── Candidate validation and family FDR ────────────────────────────────────


def _make_candidate(snr=20.0, period=5.0):
    return TransitCandidate(
        candidate_id=f"c{snr:.0f}", source_id="TIC_TEST_001",
        period_days=period, epoch_bjd=0.05, duration_hours=2.4,
        depth_ppm=5000.0, snr=snr, n_transits=5, duty_cycle=0.02,
    )


def test_validate_candidate_strong_signal(demo_lc):
    features = extract_features(demo_lc, min_period_days=0.5, max_period_days=20.0)
    cand = _make_candidate(snr=features.bls_snr)
    cand.duration_hours = features.bls_duration_hours or 2.4
    cand.period_days = features.bls_period_days or 5.0
    cand.epoch_bjd = features.bls_epoch_bjd or 0.05
    report = validate_candidate(cand, demo_lc, n_null_trials=25, max_period_days=20.0)
    assert report.verdict == "robust"
    assert report.robust
    assert report.null_result.significant_at
    assert report.holdout.agrees


def test_validate_candidate_weak_signal(flat_lc):
    report = validate_candidate(_make_candidate(snr=5.0), flat_lc, n_null_trials=12, max_period_days=15.0)
    assert report.verdict == "weak"
    assert not report.robust
    assert report.reasons


def test_family_fdr_rejects_all_when_all_significant(demo_lc):
    features = extract_features(demo_lc, min_period_days=0.5, max_period_days=20.0)
    reports = []
    for i in range(2):
        cand = _make_candidate(snr=features.bls_snr)
        cand.candidate_id = f"strong{i}"
        cand.duration_hours = features.bls_duration_hours or 2.4
        cand.period_days = features.bls_period_days or 5.0
        cand.epoch_bjd = features.bls_epoch_bjd or 0.05
        reports.append(
            validate_candidate(cand, demo_lc, n_null_trials=40, max_period_days=20.0)
        )
    apply_family_fdr(reports, alpha=0.05, method="BY")
    assert all(r.fdr is not None for r in reports)
    assert all(r.fdr.n_rejected == 2 for r in reports)


def test_by_cannot_reject_below_p_floor(demo_lc):
    """Pin the resolution limit: BY needs enough trials to reject anything.

    With m tests the largest BY threshold is alpha/C(m). At m=2 that is
    0.0333, so a p-value floor of 1/(1+n_trials) needs n >= 29 trials before a
    perfectly significant detection can be rejected at all. This is a property
    of the surrogate resolution, not of the signal, and it is the reason
    ``n_trials`` must be reported alongside every p-value.
    """
    features = extract_features(demo_lc, min_period_days=0.5, max_period_days=20.0)
    reports = []
    for i in range(2):
        cand = _make_candidate(snr=features.bls_snr)
        cand.candidate_id = f"lowres{i}"
        cand.duration_hours = features.bls_duration_hours or 2.4
        cand.period_days = features.bls_period_days or 5.0
        cand.epoch_bjd = features.bls_epoch_bjd or 0.05
        reports.append(
            validate_candidate(cand, demo_lc, n_null_trials=25, max_period_days=20.0)
        )

    # Detection is unambiguous for both...
    for r in reports:
        assert r.null_result.p_value == pytest.approx(r.null_result.p_value_floor)
        assert r.null_result.n_exceeding == 0
    # ...but at m=2 the BY threshold is 0.0333 < the 0.0385 floor, so the
    # family correction cannot express either one.
    apply_family_fdr(reports, alpha=0.05, method="BY")
    assert reports[0].fdr.n_rejected == 0

    # BH, which does not divide by C(m), rejects the same family — the two
    # procedures differ exactly where the dependence assumption matters.
    reports_bh = []
    for i in range(2):
        cand = _make_candidate(snr=features.bls_snr)
        cand.candidate_id = f"lowres{i}"
        cand.duration_hours = features.bls_duration_hours or 2.4
        cand.period_days = features.bls_period_days or 5.0
        cand.epoch_bjd = features.bls_epoch_bjd or 0.05
        reports_bh.append(
            validate_candidate(cand, demo_lc, n_null_trials=25, max_period_days=20.0)
        )
    apply_family_fdr(reports_bh, alpha=0.05, method="BH")
    assert reports_bh[0].fdr.n_rejected == 2


def test_family_fdr_rejects_none_when_none_significant(flat_lc):
    reports = [
        validate_candidate(_make_candidate(snr=5.0), flat_lc, n_null_trials=10, max_period_days=15.0)
        for _ in range(3)
    ]
    apply_family_fdr(reports, alpha=0.05, method="BY")
    assert all(r.fdr.n_rejected == 0 for r in reports)
    assert all(r.verdict == "weak" for r in reports)


def test_family_fdr_excludes_unusable_reports(demo_lc):
    features = extract_features(demo_lc, min_period_days=0.5, max_period_days=20.0)
    strong = _make_candidate(snr=features.bls_snr)
    strong.candidate_id = "only"
    strong.duration_hours = features.bls_duration_hours or 2.4
    strong.period_days = features.bls_period_days or 5.0
    strong.epoch_bjd = features.bls_epoch_bjd or 0.05
    reports = [
        validate_candidate(strong, demo_lc, n_null_trials=25, max_period_days=20.0)
    ]
    # Attach a report with no usable null
    broken = reports[0]
    broken.null_result = NullDistribution(
        observed_snr=1.0, n_trials=0, null_snr=np.array([]), p_value=1.0,
        p_value_floor=1.0, n_exceeding=0, grid_ratio=0.2, seed=0,
    )
    apply_family_fdr(reports, alpha=0.05)
    assert reports[0].fdr is None


# ── Persistence ───────────────────────────────────────────────────────────


def test_validation_roundtrip(db, demo_lc):
    features = extract_features(demo_lc, min_period_days=0.5, max_period_days=20.0)
    cand = _make_candidate(snr=features.bls_snr)
    cand.candidate_id = "roundtrip"
    cand.duration_hours = features.bls_duration_hours or 2.4
    cand.period_days = features.bls_period_days or 5.0
    cand.epoch_bjd = features.bls_epoch_bjd or 0.05
    report = validate_candidate(cand, demo_lc, n_null_trials=25, max_period_days=20.0)
    db.upsert_validation(report)
    stored = db.get_validation(report.candidate_id)
    assert stored is not None
    assert stored["verdict"] == "robust"
    assert stored["p_value"] == pytest.approx(report.null_result.p_value)
    assert stored["holdout_agrees"] == 1


def test_validation_count_by_verdict(db, demo_lc):
    for snr in (30.0, 4.0):
        r = validate_candidate(_make_candidate(snr=snr), demo_lc, n_null_trials=8, max_period_days=15.0)
        db.upsert_validation(r)
    counts = db.get_validation_count()
    assert counts.get("robust", 0) + counts.get("weak", 0) == 2


def test_review_queue_roundtrip(db):
    db.set_review_state("cand1", "pending", reviewer="nathan", note="check this")
    row = db.get_review_state("cand1")
    assert row == "pending"
    entries = db.get_review_queue(state="pending")
    assert len(entries) == 1
    assert entries[0]["reviewer"] == "nathan"


def test_review_queue_update(db):
    db.set_review_state("cand1", "pending")
    db.set_review_state("cand1", "approved", reviewer="nathan")
    assert db.get_review_state("cand1") == "approved"


def test_lightcurve_array_roundtrip(db, demo_lc, tmp_path):
    path = str(tmp_path / "lc.npz")
    db.store_lightcurve_array(demo_lc, path)
    loaded = db.load_lightcurve_array("TIC_TEST_001")
    assert loaded is not None
    assert len(loaded["time"]) == demo_lc.n_points
    assert np.allclose(loaded["time"], demo_lc.time)


def test_load_missing_lightcurve_returns_none(db):
    assert db.load_lightcurve_array("nonexistent") is None