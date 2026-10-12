"""SQLite persistence layer for Cosmic Discovery Engine."""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Optional

import numpy as np

from .models import (
    CandidateStatus,
    CrossMatch,
    DataSource,
    LightCurve,
    PipelineStage,
    StellarSource,
    TransitCandidate,
    CatalogMatch,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
    source_id TEXT PRIMARY KEY,
    ra_deg REAL NOT NULL,
    dec_deg REAL NOT NULL,
    name TEXT,
    tic_id INTEGER,
    gaia_id TEXT,
    magnitude REAL,
    teff REAL,
    radius REAL,
    mass REAL,
    metadata_json TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS light_curves (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL,
    source TEXT NOT NULL,
    sector INTEGER,
    cadence_seconds REAL,
    n_points INTEGER,
    baseline_days REAL,
    median_flux REAL,
    rms_ppm REAL,
    retrieved_at TEXT,
    checksum TEXT,
    metadata_json TEXT,
    UNIQUE(source_id, source, sector)
);

CREATE TABLE IF NOT EXISTS candidates (
    candidate_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    period_days REAL NOT NULL,
    epoch_bjd REAL NOT NULL,
    duration_hours REAL NOT NULL,
    depth_ppm REAL NOT NULL,
    snr REAL NOT NULL,
    n_transits INTEGER,
    duty_cycle REAL,
    even_odd_depth_ratio REAL,
    status TEXT NOT NULL DEFAULT 'new',
    score REAL DEFAULT 0.0,
    notes TEXT,
    false_positive_flags_json TEXT,
    metadata_json TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS cross_matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT NOT NULL,
    catalogue TEXT NOT NULL,
    match_name TEXT NOT NULL,
    match_ra_deg REAL,
    match_dec_deg REAL,
    separation_arcsec REAL,
    match_metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS catalog_objects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    catalogue TEXT NOT NULL,
    name TEXT NOT NULL,
    ra_deg REAL NOT NULL,
    dec_deg REAL NOT NULL,
    period_days REAL,
    epoch_bjd REAL,
    depth_ppm REAL,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS validations (
    candidate_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    verdict TEXT NOT NULL,
    p_value REAL,
    p_value_floor REAL,
    n_null_trials INTEGER,
    n_null_exceeding INTEGER,
    grid_ratio REAL,
    null_seed INTEGER,
    holdout_agrees INTEGER,
    holdout_segment_b_snr REAL,
    holdout_n_transits_observed INTEGER,
    holdout_verdict TEXT,
    sensitivity_survival_rate REAL,
    sensitivity_verdict TEXT,
    fdr_method TEXT,
    fdr_alpha REAL,
    fdr_n_tests INTEGER,
    fdr_rejected INTEGER,
    reasons_json TEXT,
    report_json TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS review_queue (
    candidate_id TEXT PRIMARY KEY,
    state TEXT NOT NULL DEFAULT 'pending',
    reviewer TEXT,
    note TEXT,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS light_curve_data (
    source_id TEXT NOT NULL,
    source TEXT NOT NULL,
    sector INTEGER,
    time_min REAL NOT NULL,
    time_max REAL NOT NULL,
    n_points INTEGER NOT NULL,
    path TEXT NOT NULL,
    PRIMARY KEY (source_id, source, sector)
);

CREATE INDEX IF NOT EXISTS idx_validations_verdict ON validations(verdict);
CREATE INDEX IF NOT EXISTS idx_validations_pvalue ON validations(p_value);
CREATE INDEX IF NOT EXISTS idx_candidates_source ON candidates(source_id);
CREATE INDEX IF NOT EXISTS idx_candidates_status ON candidates(status);
CREATE INDEX IF NOT EXISTS idx_candidates_score ON candidates(score DESC);
CREATE INDEX IF NOT EXISTS idx_cross_matches_candidate ON cross_matches(candidate_id);
CREATE INDEX IF NOT EXISTS idx_catalog_ra ON catalog_objects(ra_deg);
CREATE INDEX IF NOT EXISTS idx_catalog_dec ON catalog_objects(dec_deg);
"""


class Database:
    """Thin SQLite wrapper with typed helpers."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            conn.execute(
                "INSERT OR REPLACE INTO schema_meta (key, value) VALUES (?, ?)",
                ("version", str(SCHEMA_VERSION)),
            )

    # ── Sources ──────────────────────────────────────────────────────────

    def upsert_source(self, src: StellarSource) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sources (source_id, ra_deg, dec_deg, name, tic_id, gaia_id,
                                     magnitude, teff, radius, mass, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id) DO UPDATE SET
                    ra_deg=excluded.ra_deg, dec_deg=excluded.dec_deg,
                    name=COALESCE(excluded.name, sources.name),
                    tic_id=COALESCE(excluded.tic_id, sources.tic_id),
                    gaia_id=COALESCE(excluded.gaia_id, sources.gaia_id),
                    magnitude=COALESCE(excluded.magnitude, sources.magnitude),
                    teff=COALESCE(excluded.teff, sources.teff),
                    radius=COALESCE(excluded.radius, sources.radius),
                    mass=COALESCE(excluded.mass, sources.mass),
                    metadata_json=COALESCE(excluded.metadata_json, sources.metadata_json)
                """,
                (
                    src.source_id, src.ra_deg, src.dec_deg, src.name,
                    src.tic_id, src.gaia_id, src.magnitude, src.teff,
                    src.radius, src.mass, json.dumps(src.metadata),
                ),
            )

    def get_source(self, source_id: str) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sources WHERE source_id = ?", (source_id,)
            ).fetchone()
            return dict(row) if row else None

    # ── Light curves ────────────────────────────────────────────────────

    def upsert_lightcurve(self, lc: LightCurve) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO light_curves (source_id, source, sector, cadence_seconds,
                                          n_points, baseline_days, median_flux, rms_ppm,
                                          retrieved_at, checksum, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id, source, sector) DO UPDATE SET
                    cadence_seconds=excluded.cadence_seconds,
                    n_points=excluded.n_points,
                    baseline_days=excluded.baseline_days,
                    median_flux=excluded.median_flux,
                    rms_ppm=excluded.rms_ppm,
                    retrieved_at=excluded.retrieved_at,
                    checksum=excluded.checksum,
                    metadata_json=excluded.metadata_json
                """,
                (
                    lc.source_id, lc.source.value, lc.sector, lc.cadence_seconds,
                    lc.n_points, lc.baseline_days, lc.median_flux, lc.rms_ppm,
                    lc.retrieved_at.isoformat(), lc.checksum,
                    json.dumps(lc.metadata),
                ),
            )
            return cur.lastrowid or 0

    def get_lightcurve_meta(self, source_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM light_curves WHERE source_id = ? ORDER BY source, sector",
                (source_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ── Light curve arrays (for dashboard plotting) ────────────────────

    def store_lightcurve_array(self, lc: Any, path: str) -> None:
        """Write the time/flux arrays to a compressed npz and index it."""
        np.savez_compressed(
            path,
            time=np.asarray(lc.time, dtype=np.float64),
            flux=np.asarray(lc.flux, dtype=np.float64),
            flux_err=np.asarray(lc.flux_err, dtype=np.float64),
        )
        sector = lc.sector if lc.sector is not None else -1
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO light_curve_data (source_id, source, sector, time_min, time_max, n_points, path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id, source, sector) DO UPDATE SET
                    time_min=excluded.time_min, time_max=excluded.time_max,
                    n_points=excluded.n_points, path=excluded.path
                """,
                (
                    lc.source_id, lc.source.value, lc.sector,
                    float(lc.time.min()), float(lc.time.max()), lc.n_points, path,
                ),
            )

    def load_lightcurve_array(
        self, source_id: str, source: Optional[str] = None
    ) -> Optional[dict[str, Any]]:
        """Load a stored light curve. Returns None when nothing is indexed."""
        q = "SELECT * FROM light_curve_data WHERE source_id = ?"
        params: list[Any] = [source_id]
        if source:
            q += " AND source = ?"
            params.append(source)
        q += " ORDER BY n_points DESC LIMIT 1"
        with self._connect() as conn:
            row = conn.execute(q, params).fetchone()
        if row is None:
            return None
        data_path = Path(row["path"])
        if not data_path.exists():
            return None
        arrays = np.load(str(data_path))
        return {
            "time": arrays["time"],
            "flux": arrays["flux"],
            "flux_err": arrays["flux_err"],
            "source": row["source"],
            "sector": row["sector"],
        }

    def get_review_state(self, candidate_id: str) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT state FROM review_queue WHERE candidate_id = ?", (candidate_id,)
            ).fetchone()
            return row["state"] if row else None

    # ── Candidates ──────────────────────────────────────────────────────

    def upsert_candidate(self, cand: TransitCandidate) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO candidates (candidate_id, source_id, period_days, epoch_bjd,
                                        duration_hours, depth_ppm, snr, n_transits,
                                        duty_cycle, even_odd_depth_ratio, status, score,
                                        notes, false_positive_flags_json, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(candidate_id) DO UPDATE SET
                    period_days=excluded.period_days,
                    epoch_bjd=excluded.epoch_bjd,
                    duration_hours=excluded.duration_hours,
                    depth_ppm=excluded.depth_ppm,
                    snr=excluded.snr,
                    n_transits=excluded.n_transits,
                    duty_cycle=excluded.duty_cycle,
                    even_odd_depth_ratio=excluded.even_odd_depth_ratio,
                    status=excluded.status,
                    score=excluded.score,
                    notes=excluded.notes,
                    false_positive_flags_json=excluded.false_positive_flags_json,
                    metadata_json=excluded.metadata_json
                """,
                (
                    cand.candidate_id, cand.source_id, cand.period_days,
                    cand.epoch_bjd, cand.duration_hours, cand.depth_ppm,
                    cand.snr, cand.n_transits, cand.duty_cycle,
                    cand.even_odd_depth_ratio, cand.status.value, cand.score,
                    cand.notes, json.dumps(cand.false_positive_flags),
                    json.dumps(cand.metadata),
                ),
            )

    def get_candidates(
        self,
        status: Optional[CandidateStatus] = None,
        min_score: float = 0.0,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        q = "SELECT * FROM candidates WHERE score >= ?"
        params: list[Any] = [min_score]
        if status is not None:
            q += " AND status = ?"
            params.append(status.value)
        q += " ORDER BY score DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(q, params).fetchall()
            return [dict(r) for r in rows]

    def get_candidate_count(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) as cnt FROM candidates GROUP BY status"
            ).fetchall()
            return {r["status"]: r["cnt"] for r in rows}

    def update_candidate_status(self, candidate_id: str, status: CandidateStatus) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE candidates SET status = ? WHERE candidate_id = ?",
                (status.value, candidate_id),
            )

    # ── Cross-matches ───────────────────────────────────────────────────

    def upsert_crossmatch(self, cm: CrossMatch) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cross_matches (candidate_id, catalogue, match_name,
                                           match_ra_deg, match_dec_deg,
                                           separation_arcsec, match_metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cm.candidate_id, cm.catalogue, cm.match_name,
                    cm.match_ra_deg, cm.match_dec_deg, cm.separation_arcsec,
                    json.dumps(cm.match_metadata),
                ),
            )

    def get_crossmatches(self, candidate_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM cross_matches WHERE candidate_id = ? ORDER BY separation_arcsec",
                (candidate_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ── Validations ─────────────────────────────────────────────────────

    def upsert_validation(self, report: Any) -> None:
        """Persist a ValidationReport. Accepts the dataclass directly."""
        null = report.null_result
        hold = report.holdout
        sens = report.sensitivity
        fdr = report.fdr
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO validations (
                    candidate_id, source_id, verdict, p_value, p_value_floor,
                    n_null_trials, n_null_exceeding, grid_ratio, null_seed,
                    holdout_agrees, holdout_segment_b_snr, holdout_n_transits_observed,
                    holdout_verdict, sensitivity_survival_rate, sensitivity_verdict,
                    fdr_method, fdr_alpha, fdr_n_tests, fdr_rejected,
                    reasons_json, report_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(candidate_id) DO UPDATE SET
                    verdict=excluded.verdict, p_value=excluded.p_value,
                    p_value_floor=excluded.p_value_floor,
                    n_null_trials=excluded.n_null_trials,
                    n_null_exceeding=excluded.n_null_exceeding,
                    grid_ratio=excluded.grid_ratio, null_seed=excluded.null_seed,
                    holdout_agrees=excluded.holdout_agrees,
                    holdout_segment_b_snr=excluded.holdout_segment_b_snr,
                    holdout_n_transits_observed=excluded.holdout_n_transits_observed,
                    holdout_verdict=excluded.holdout_verdict,
                    sensitivity_survival_rate=excluded.sensitivity_survival_rate,
                    sensitivity_verdict=excluded.sensitivity_verdict,
                    fdr_method=excluded.fdr_method, fdr_alpha=excluded.fdr_alpha,
                    fdr_n_tests=excluded.fdr_n_tests, fdr_rejected=excluded.fdr_rejected,
                    reasons_json=excluded.reasons_json, report_json=excluded.report_json
                """,
                (
                    report.candidate_id, report.source_id, report.verdict,
                    null.p_value if null else None,
                    null.p_value_floor if null else None,
                    null.n_trials if null else None,
                    null.n_exceeding if null else None,
                    null.grid_ratio if null else None,
                    null.seed if null else None,
                    int(bool(hold.agrees)) if hold else None,
                    hold.segment_b_snr if hold else None,
                    hold.n_transits_observed if hold else None,
                    hold.verdict if hold else None,
                    sens.survival_rate if sens else None,
                    sens.verdict if sens else None,
                    fdr.method if fdr else None,
                    fdr.alpha if fdr else None,
                    fdr.n_tests if fdr else None,
                    int(report.candidate_id in set(fdr.rejected)) if fdr else None,
                    json.dumps(report.reasons),
                    json.dumps(report.to_dict(), default=str),
                ),
            )

    def get_validation(self, candidate_id: str) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM validations WHERE candidate_id = ?", (candidate_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_validations(self, verdict: Optional[str] = None, limit: int = 500) -> list[dict[str, Any]]:
        q = "SELECT * FROM validations"
        params: list[Any] = []
        if verdict:
            q += " WHERE verdict = ?"
            params.append(verdict)
        q += " ORDER BY p_value ASC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(q, params).fetchall()]

    def get_validation_count(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT verdict, COUNT(*) as cnt FROM validations GROUP BY verdict"
            ).fetchall()
            return {r["verdict"]: r["cnt"] for r in rows}

    # ── Review queue ────────────────────────────────────────────────────

    def set_review_state(
        self, candidate_id: str, state: str, reviewer: str = "", note: str = ""
    ) -> None:
        """Move a candidate into the human review queue."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO review_queue (candidate_id, state, reviewer, note, updated_at)
                VALUES (?, ?, ?, ?, datetime('now'))
                ON CONFLICT(candidate_id) DO UPDATE SET
                    state=excluded.state, reviewer=excluded.reviewer,
                    note=excluded.note, updated_at=excluded.updated_at
                """,
                (candidate_id, state, reviewer, note),
            )

    def get_review_queue(self, state: Optional[str] = None, limit: int = 200) -> list[dict[str, Any]]:
        q = """
            SELECT r.*, c.source_id, c.period_days, c.depth_ppm, c.snr, c.score, c.status
            FROM review_queue r LEFT JOIN candidates c ON c.candidate_id = r.candidate_id
        """
        params: list[Any] = []
        if state:
            q += " WHERE r.state = ?"
            params.append(state)
        q += " ORDER BY r.updated_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(q, params).fetchall()]

    # ── Catalog objects ─────────────────────────────────────────────────

    def upsert_catalog_object(self, obj: CatalogMatch) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO catalog_objects (catalogue, name, ra_deg, dec_deg,
                                             period_days, epoch_bjd, depth_ppm, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    obj.catalogue, obj.name, obj.ra_deg, obj.dec_deg,
                    obj.period_days, obj.epoch_bjd, obj.depth_ppm,
                    json.dumps(obj.metadata),
                ),
            )

    def search_catalog(
        self,
        ra_deg: float,
        dec_deg: float,
        radius_arcsec: float,
        catalogue: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Simple cone search using pre-filter + exact angular distance."""
        # Rough pre-filter: 1 degree ~ 3600 arcsec, use bounding box
        radius_deg = radius_arcsec / 3600.0
        q = "SELECT * FROM catalog_objects WHERE ra_deg BETWEEN ? AND ? AND dec_deg BETWEEN ? AND ?"
        params: list[Any] = [
            ra_deg - radius_deg, ra_deg + radius_deg,
            dec_deg - radius_deg, dec_deg + radius_deg,
        ]
        if catalogue:
            q += " AND catalogue = ?"
            params.append(catalogue)
        with self._connect() as conn:
            rows = conn.execute(q, params).fetchall()
            results = []
            for r in rows:
                d = _angular_sep(ra_deg, dec_deg, r["ra_deg"], r["dec_deg"])
                if d <= radius_arcsec:
                    results.append(dict(r, separation_arcsec=d))
            return results


def _angular_sep(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Angular separation in arcsec using the Haversine formula."""
    import math
    r1, d1, r2, d2 = map(math.radians, [ra1, dec1, ra2, dec2])
    dra = r2 - r1
    ddec = d2 - d1
    a = math.sin(ddec / 2) ** 2 + math.cos(d1) * math.cos(d2) * math.sin(dra / 2) ** 2
    return math.degrees(2 * math.asin(math.sqrt(a))) * 3600.0
