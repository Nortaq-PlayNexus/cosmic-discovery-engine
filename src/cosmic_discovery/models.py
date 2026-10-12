"""Typed data models for astronomical entities."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
from typing import Any, Optional
import numpy as np


class DataSource(str, Enum):
    MAST = "mast"
    EXOPLANET_ARCHIVE = "exoplanet_archive"
    GAIA = "gaia"
    DEMO = "demo"


class CandidateStatus(str, Enum):
    NEW = "new"
    KNOWN = "known"
    FALSE_POSITIVE = "false_positive"
    UNVERIFIED = "unverified"


class PipelineStage(str, Enum):
    INGESTED = "ingested"
    CLEANED = "cleaned"
    FEATURED = "featured"
    CANDIDATE = "candidate"
    VALIDATED = "validated"
    CROSS_MATCHED = "cross_matched"


@dataclass
class StellarSource:
    """A stellar target with cross-identifications."""

    source_id: str
    ra_deg: float
    dec_deg: float
    name: Optional[str] = None
    tic_id: Optional[int] = None
    gaia_id: Optional[str] = None
    magnitude: Optional[float] = None
    teff: Optional[float] = None
    radius: Optional[float] = None
    mass: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "ra_deg": self.ra_deg,
            "dec_deg": self.dec_deg,
            "name": self.name,
            "tic_id": self.tic_id,
            "gaia_id": self.gaia_id,
            "magnitude": self.magnitude,
            "teff": self.teff,
            "radius": self.radius,
            "mass": self.mass,
        }


@dataclass
class LightCurve:
    """A time-series photometry dataset."""

    source_id: str
    time: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    source: DataSource
    sector: Optional[int] = None
    cadence_seconds: float = 120.0
    quality_flags: Optional[np.ndarray] = None
    retrieved_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    checksum: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.time)
        if len(self.flux) != n or len(self.flux_err) != n:
            raise ValueError("time, flux, flux_err must have equal length")

    @property
    def n_points(self) -> int:
        return len(self.time)

    @property
    def baseline_days(self) -> float:
        return float(self.time[-1] - self.time[0]) if self.n_points > 1 else 0.0

    @property
    def median_flux(self) -> float:
        return float(np.nanmedian(self.flux))

    @property
    def rms_ppm(self) -> float:
        """RMS scatter in parts-per-million."""
        clean = self.flux[np.isfinite(self.flux)]
        if len(clean) < 2:
            return 0.0
        med = np.median(clean)
        if med == 0:
            return 0.0
        return float(np.std(clean) / med * 1e6)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source": self.source.value,
            "sector": self.sector,
            "cadence_seconds": self.cadence_seconds,
            "n_points": self.n_points,
            "baseline_days": self.baseline_days,
            "median_flux": self.median_flux,
            "rms_ppm": self.rms_ppm,
            "retrieved_at": self.retrieved_at.isoformat(),
            "checksum": self.checksum,
        }


@dataclass
class TransitCandidate:
    """A detected transit-like signal."""

    candidate_id: str
    source_id: str
    period_days: float
    epoch_bjd: float
    duration_hours: float
    depth_ppm: float
    snr: float
    n_transits: int
    duty_cycle: float
    even_odd_depth_ratio: Optional[float] = None
    centroid_offset: Optional[float] = None
    false_positive_flags: list[str] = field(default_factory=list)
    status: CandidateStatus = CandidateStatus.NEW
    score: float = 0.0
    notes: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "source_id": self.source_id,
            "period_days": self.period_days,
            "epoch_bjd": self.epoch_bjd,
            "duration_hours": self.duration_hours,
            "depth_ppm": self.depth_ppm,
            "snr": self.snr,
            "n_transits": self.n_transits,
            "duty_cycle": self.duty_cycle,
            "even_odd_depth_ratio": self.even_odd_depth_ratio,
            "status": self.status.value,
            "score": self.score,
            "notes": self.notes,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class CrossMatch:
    """Cross-match result between a candidate and a known catalogue."""

    candidate_id: str
    catalogue: str
    match_name: str
    match_ra_deg: float
    match_dec_deg: float
    separation_arcsec: float
    match_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "catalogue": self.catalogue,
            "match_name": self.match_name,
            "separation_arcsec": self.separation_arcsec,
        }


@dataclass
class CatalogMatch:
    """A known object from a catalogue that may match a candidate."""

    catalogue: str
    name: str
    ra_deg: float
    dec_deg: float
    period_days: Optional[float] = None
    epoch_bjd: Optional[float] = None
    depth_ppm: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)
