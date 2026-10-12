"""Tests for data models."""

from __future__ import annotations

import numpy as np
import pytest

from cosmic_discovery.models import (
    CandidateStatus,
    DataSource,
    LightCurve,
    PipelineStage,
    StellarSource,
    TransitCandidate,
)


def test_stellar_source_creation():
    src = StellarSource(
        source_id="TIC 123",
        ra_deg=180.0,
        dec_deg=-30.0,
        tic_id=123,
        magnitude=10.5,
    )
    d = src.to_dict()
    assert d["source_id"] == "TIC 123"
    assert d["tic_id"] == 123
    assert d["magnitude"] == 10.5


def test_lightcurve_creation():
    n = 100
    lc = LightCurve(
        source_id="TIC 456",
        time=np.arange(n, dtype=float),
        flux=np.ones(n),
        flux_err=np.full(n, 0.001),
        source=DataSource.DEMO,
    )
    assert lc.n_points == n
    assert lc.baseline_days == 99.0
    assert lc.median_flux == pytest.approx(1.0)
    assert lc.rms_ppm < 100.0  # very low RMS for flat light


def test_lightcurve_length_mismatch():
    with pytest.raises(ValueError, match="equal length"):
        LightCurve(
            source_id="bad",
            time=np.arange(10),
            flux=np.ones(5),
            flux_err=np.ones(10),
            source=DataSource.DEMO,
        )


def test_transit_candidate_creation():
    cand = TransitCandidate(
        candidate_id="abc123",
        source_id="TIC 789",
        period_days=5.0,
        epoch_bjd=2457000.0,
        duration_hours=2.5,
        depth_ppm=5000.0,
        snr=10.0,
        n_transits=5,
        duty_cycle=0.02,
    )
    d = cand.to_dict()
    assert d["status"] == "new"
    assert d["score"] == 0.0


def test_candidate_status_enum():
    assert CandidateStatus.NEW == "new"
    assert CandidateStatus.KNOWN == "known"
    assert CandidateStatus.FALSE_POSITIVE == "false_positive"
    assert CandidateStatus.UNVERIFIED == "unverified"


def test_data_source_enum():
    assert DataSource.MAST == "mast"
    assert DataSource.DEMO == "demo"
