"""Cross-match candidates against known catalogues."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from ..models import (
    CandidateStatus,
    CatalogMatch,
    CrossMatch,
    TransitCandidate,
)
from ..database import Database

logger = logging.getLogger(__name__)


@dataclass
class CrossMatchResult:
    """Result of cross-matching a candidate against catalogues."""

    candidate: TransitCandidate
    matches: list[CrossMatch] = field(default_factory=list)
    is_known: bool = False
    known_as: Optional[str] = None
    period_match: bool = False
    notes: str = ""


def crossmatch_candidate(
    candidate: TransitCandidate,
    db: Database,
    search_radius_arcsec: float = 10.0,
) -> CrossMatchResult:
    """Cross-match a candidate against all catalogues in the database.

    Strategy:
    1. Look up the source position from the sources table.
    2. Search catalog_objects within the search radius.
    3. If a match is found, compare period and epoch.
    4. Update candidate status accordingly.
    """
    # Get source position
    source = db.get_source(candidate.source_id)
    if source is None:
        return CrossMatchResult(
            candidate=candidate,
            notes="Source not in database; cannot cross-match",
        )

    ra = source["ra_deg"]
    dec = source["dec_deg"]

    # Search all catalogues
    catalog_hits = db.search_catalog(ra, dec, search_radius_arcsec)

    if not catalog_hits:
        return CrossMatchResult(
            candidate=candidate,
            matches=[],
            is_known=False,
            notes="No catalog matches within search radius",
        )

    matches: list[CrossMatch] = []
    is_known = False
    known_as: Optional[str] = None
    period_match = False

    for hit in catalog_hits:
        cm = CrossMatch(
            candidate_id=candidate.candidate_id,
            catalogue=hit["catalogue"],
            match_name=hit["name"],
            match_ra_deg=hit["ra_deg"],
            match_dec_deg=hit["dec_deg"],
            separation_arcsec=hit["separation_arcsec"],
            match_metadata={
                "catalog_period_days": hit.get("period_days"),
                "catalog_epoch_bjd": hit.get("epoch_bjd"),
                "catalog_depth_ppm": hit.get("depth_ppm"),
            },
        )
        matches.append(cm)
        db.upsert_crossmatch(cm)

        # Check if this is a period match
        cat_period = hit.get("period_days")
        if cat_period and candidate.period_days:
            # Within 1% or within 10 seconds
            period_diff_pct = abs(cat_period - candidate.period_days) / cat_period * 100
            if period_diff_pct < 1.0:
                period_match = True
                is_known = True
                known_as = hit["name"]

    # Update candidate status
    if is_known:
        candidate.status = CandidateStatus.KNOWN
        candidate.notes += f"; Known as {known_as} (period match)"
    elif len(matches) > 0:
        candidate.status = CandidateStatus.UNVERIFIED
        candidate.notes += f"; {len(matches)} catalog match(es) but no period match"
    else:
        candidate.notes += "; No catalog matches"

    # Update in database
    db.upsert_candidate(candidate)

    return CrossMatchResult(
        candidate=candidate,
        matches=matches,
        is_known=is_known,
        known_as=known_as,
        period_match=period_match,
    )
