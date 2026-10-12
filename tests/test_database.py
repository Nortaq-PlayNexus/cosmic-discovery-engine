"""Tests for the database layer."""

from __future__ import annotations

import pytest

from cosmic_discovery.models import (
    CandidateStatus,
    CrossMatch,
    DataSource,
    LightCurve,
    StellarSource,
    TransitCandidate,
    CatalogMatch,
)
from cosmic_discovery.database import Database, _angular_sep
import numpy as np


def test_database_init(db):
    """Database should initialize with schema."""
    assert db.db_path.exists()


def test_source_upsert_and_get(db):
    src = StellarSource(
        source_id="TIC 100",
        ra_deg=10.0,
        dec_deg=20.0,
        name="Test Star",
        tic_id=100,
    )
    db.upsert_source(src)

    retrieved = db.get_source("TIC 100")
    assert retrieved is not None
    assert retrieved["name"] == "Test Star"
    assert retrieved["tic_id"] == 100


def test_source_upsert_update(db):
    """Upsert should update existing source."""
    src1 = StellarSource(source_id="TIC 200", ra_deg=30.0, dec_deg=40.0, magnitude=12.0)
    db.upsert_source(src1)

    src2 = StellarSource(source_id="TIC 200", ra_deg=30.0, dec_deg=40.0, magnitude=11.5)
    db.upsert_source(src2)

    retrieved = db.get_source("TIC 200")
    assert retrieved["magnitude"] == 11.5


def test_lightcurve_upsert(db):
    lc = LightCurve(
        source_id="TIC 300",
        time=np.arange(10, dtype=float),
        flux=np.ones(10),
        flux_err=np.full(10, 0.001),
        source=DataSource.DEMO,
        sector=1,
    )
    row_id = db.upsert_lightcurve(lc)
    assert row_id > 0

    meta = db.get_lightcurve_meta("TIC 300")
    assert len(meta) == 1
    assert meta[0]["sector"] == 1


def test_candidate_upsert_and_query(db):
    cand = TransitCandidate(
        candidate_id="cand001",
        source_id="TIC 400",
        period_days=3.5,
        epoch_bjd=2458000.0,
        duration_hours=2.0,
        depth_ppm=3000.0,
        snr=8.0,
        n_transits=4,
        duty_cycle=0.023,
        score=65.0,
    )
    db.upsert_candidate(cand)

    results = db.get_candidates(min_score=50.0)
    assert len(results) == 1
    assert results[0]["candidate_id"] == "cand001"
    assert results[0]["score"] == 65.0


def test_candidate_status_filter(db):
    c1 = TransitCandidate(
        candidate_id="c1", source_id="S1",
        period_days=1.0, epoch_bjd=0.0, duration_hours=1.0,
        depth_ppm=1000.0, snr=10.0, n_transits=3, duty_cycle=0.01,
        status=CandidateStatus.NEW, score=80.0,
    )
    c2 = TransitCandidate(
        candidate_id="c2", source_id="S2",
        period_days=2.0, epoch_bjd=0.0, duration_hours=2.0,
        depth_ppm=2000.0, snr=12.0, n_transits=5, duty_cycle=0.02,
        status=CandidateStatus.KNOWN, score=90.0,
    )
    db.upsert_candidate(c1)
    db.upsert_candidate(c2)

    new_only = db.get_candidates(status=CandidateStatus.NEW)
    assert len(new_only) == 1
    assert new_only[0]["candidate_id"] == "c1"

    known_only = db.get_candidates(status=CandidateStatus.KNOWN)
    assert len(known_only) == 1


def test_candidate_count(db):
    c = TransitCandidate(
        candidate_id="cc1", source_id="S",
        period_days=1.0, epoch_bjd=0.0, duration_hours=1.0,
        depth_ppm=500.0, snr=5.0, n_transits=2, duty_cycle=0.01,
    )
    db.upsert_candidate(c)
    counts = db.get_candidate_count()
    assert counts.get("new") == 1


def test_update_candidate_status(db):
    c = TransitCandidate(
        candidate_id="st1", source_id="S",
        period_days=1.0, epoch_bjd=0.0, duration_hours=1.0,
        depth_ppm=500.0, snr=5.0, n_transits=2, duty_cycle=0.01,
    )
    db.upsert_candidate(c)
    db.update_candidate_status("st1", CandidateStatus.KNOWN)

    results = db.get_candidates(status=CandidateStatus.KNOWN)
    assert len(results) == 1


def test_catalog_upsert_and_search(db):
    obj = CatalogMatch(
        catalogue="exoplanet_archive",
        name="Kepler-10 b",
        ra_deg=285.0,
        dec_deg=48.0,
        period_days=0.837,
    )
    db.upsert_catalog_object(obj)

    hits = db.search_catalog(285.0, 48.0, 30.0)
    assert len(hits) == 1
    assert hits[0]["name"] == "Kepler-10 b"
    assert hits[0]["separation_arcsec"] < 1.0


def test_catalog_search_empty(db):
    hits = db.search_catalog(0.0, 0.0, 10.0)
    assert len(hits) == 0


def test_crossmatch_upsert(db):
    cm = CrossMatch(
        candidate_id="cm1",
        catalogue="exoplanet_archive",
        match_name="Kepler-22 b",
        match_ra_deg=289.0,
        match_dec_deg=47.0,
        separation_arcsec=2.5,
    )
    db.upsert_crossmatch(cm)

    results = db.get_crossmatches("cm1")
    assert len(results) == 1
    assert results[0]["match_name"] == "Kepler-22 b"


def test_angular_sep_zero():
    sep = _angular_sep(180.0, 45.0, 180.0, 45.0)
    assert sep == pytest.approx(0.0, abs=1e-10)


def test_angular_sep_known():
    # 1 degree apart in RA at equator ~ 3600 arcsec
    sep = _angular_sep(10.0, 0.0, 11.0, 0.0)
    assert sep == pytest.approx(3600.0, rel=0.01)
