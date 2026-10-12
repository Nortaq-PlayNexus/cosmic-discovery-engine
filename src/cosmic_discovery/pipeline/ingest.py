"""Ingestion orchestrator: fetch data, extract features, generate candidates."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from ..config import load_config
from ..database import Database
from ..models import DataSource, LightCurve, StellarSource
from ..sources import DataSourceBase, ExoplanetArchiveSource, DemoSource, MASTSource, SourceResult
from .candidates import CandidateGenerator
from .features import LightCurveFeatures, extract_features
from .crossmatch import CrossMatchResult, crossmatch_candidate

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    """Result of ingesting one source."""

    source_id: str
    success: bool
    lightcurve: Optional[LightCurve] = None
    features: Optional[LightCurveFeatures] = None
    candidates: list = field(default_factory=list)
    crossmatch: Optional[CrossMatchResult] = None
    error: Optional[str] = None


class IngestPipeline:
    """End-to-end pipeline: fetch light curve, extract features, generate candidates, cross-match."""

    def __init__(
        self,
        db: Database,
        source: DataSourceBase,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        self.db = db
        self.source = source
        self.config = config or load_config()
        self.candidate_generator = CandidateGenerator(
            min_snr=self.config.get("pipeline", {}).get("min_snr", 5.0),
            min_depth_ppm=self.config.get("pipeline", {}).get("min_transit_depth_ppm", 100),
            max_depth_ppm=self.config.get("pipeline", {}).get("max_transit_depth_ppm", 50000),
            min_n_transits=2,
        )

    def process(self, source_id: str, **kwargs: Any) -> IngestResult:
        """Process a single source through the full pipeline."""
        logger.info("Processing %s through ingestion pipeline", source_id)

        # 1. Fetch light curve
        fetch_result = self.source.fetch_lightcurve(source_id, **kwargs)
        if not fetch_result.success or fetch_result.data is None:
            error = fetch_result.error or "Unknown fetch failure"
            logger.error("Failed to fetch light curve for %s: %s", source_id, error)
            return IngestResult(source_id=source_id, success=False, error=error)

        lc = fetch_result.data

        # 2. Persist source and light curve metadata
        if hasattr(self.source, 'get_source_info'):
            src_info = self.source.get_source_info(source_id)
            if src_info:
                self.db.upsert_source(src_info)
        self.db.upsert_lightcurve(lc)

        # 3. Extract features
        pipeline_cfg = self.config.get("pipeline", {})
        features = extract_features(
            lc,
            min_period_days=pipeline_cfg.get("min_period_days", 0.1),
            max_period_days=pipeline_cfg.get("max_period_days", 50.0),
            bls_duration_grid=pipeline_cfg.get("bls_duration_grid"),
            detrend_window=pipeline_cfg.get("detrend_window", 101),
            outlier_sigma=pipeline_cfg.get("outlier_sigma", 5.0),
        )

        # 4. Generate candidates
        candidate = self.candidate_generator.generate(features)
        candidates = [candidate] if candidate else []

        # 5. Persist candidates
        for cand in candidates:
            self.db.upsert_candidate(cand)

        # 6. Cross-match
        crossmatch_result = None
        for cand in candidates:
            crossmatch_result = crossmatch_candidate(
                cand,
                self.db,
                search_radius_arcsec=self.config.get("crossmatch", {}).get("search_radius_arcsec", 10.0),
            )

        return IngestResult(
            source_id=source_id,
            success=True,
            lightcurve=lc,
            features=features,
            candidates=candidates,
            crossmatch=crossmatch_result,
        )

    def process_many(self, source_ids: list[str], **kwargs: Any) -> list[IngestResult]:
        """Process multiple sources, collecting results."""
        results = []
        for sid in source_ids:
            try:
                result = self.process(sid, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error("Pipeline failed for %s: %s", sid, e)
                results.append(IngestResult(source_id=sid, success=False, error=str(e)))
        return results


def create_demo_pipeline(db_path: str | None = None) -> IngestPipeline:
    """Create a pipeline wired to the demo data source."""
    from ..config import get_db_path
    config = load_config()
    db = Database(db_path or get_db_path(config))
    source = DemoSource()
    return IngestPipeline(db=db, source=source, config=config)


def create_mast_pipeline(
    db_path: str | None = None,
    cache_dir: str | None = None,
    token: str | None = None,
) -> IngestPipeline:
    """Create a pipeline wired to the MAST data source."""
    from ..config import get_db_path
    config = load_config()
    db = Database(db_path or get_db_path(config))
    source = MASTSource(
        cache_dir=cache_dir or config.get("sources", {}).get("mast", {}).get("cache_dir"),
        token=token,
    )
    return IngestPipeline(db=db, source=source, config=config)
