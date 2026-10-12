"""Tests for the ingestion pipeline."""

from __future__ import annotations

import pytest

from cosmic_discovery.config import load_config, get_db_path
from cosmic_discovery.database import Database
from cosmic_discovery.models import CandidateStatus, StellarSource
from cosmic_discovery.pipeline.ingest import IngestPipeline, create_demo_pipeline
from cosmic_discovery.sources.demo import DemoSource


def test_demo_pipeline_creation(tmp_path):
    db_path = tmp_path / "test.db"
    pipeline = create_demo_pipeline(db_path=str(db_path))
    assert pipeline.db is not None
    assert pipeline.source is not None
    assert pipeline.source.name == "demo"


def test_pipeline_process_demo_target(tmp_path):
    db_path = tmp_path / "test.db"
    pipeline = create_demo_pipeline(db_path=str(db_path))
    result = pipeline.process("TIC 100000001")
    assert result.success
    assert result.lightcurve is not None
    assert result.features is not None


def test_pipeline_process_many(tmp_path):
    db_path = tmp_path / "test.db"
    pipeline = create_demo_pipeline(db_path=str(db_path))
    results = pipeline.process_many(["TIC 100000001", "TIC 100000002"])
    assert len(results) == 2
    assert all(r.success for r in results)


def test_pipeline_persists_to_database(tmp_path):
    db_path = tmp_path / "test.db"
    pipeline = create_demo_pipeline(db_path=str(db_path))
    pipeline.process("TIC 100000001")

    # Verify data was persisted
    db = Database(db_path)
    source = db.get_source("TIC 100000001")
    assert source is not None

    lc_meta = db.get_lightcurve_meta("TIC 100000001")
    assert len(lc_meta) > 0


def test_pipeline_ingests_source_info(tmp_path):
    db_path = tmp_path / "test.db"
    pipeline = create_demo_pipeline(db_path=str(db_path))
    pipeline.process("TIC 100000003")

    db = Database(db_path)
    source = db.get_source("TIC 100000003")
    assert source is not None
    assert "ra_deg" in source


def test_config_load_defaults():
    config = load_config()
    assert "database" in config
    assert "pipeline" in config
    assert "sources" in config


def test_config_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("CDE_PIPELINE__MIN_SNR", "7.5")
    config = load_config()
    assert config["pipeline"]["min_snr"] == 7.5
