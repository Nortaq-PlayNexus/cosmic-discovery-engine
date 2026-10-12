"""Candidate generation from extracted features."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Optional

from ..models import CandidateStatus, TransitCandidate
from .features import LightCurveFeatures, extract_features
from ..models import LightCurve

logger = logging.getLogger(__name__)


class CandidateGenerator:
    """Convert BLS-detected transit signals into scored candidates."""

    def __init__(
        self,
        min_snr: float = 5.0,
        min_depth_ppm: float = 100.0,
        max_depth_ppm: float = 50000.0,
        min_n_transits: int = 2,
    ) -> None:
        self.min_snr = min_snr
        self.min_depth_ppm = min_depth_ppm
        self.max_depth_ppm = max_depth_ppm
        self.min_n_transits = min_n_transits

    def generate(self, features: LightCurveFeatures) -> Optional[TransitCandidate]:
        """Create a transit candidate from BLS features, or None if below thresholds."""
        if features.bls_period_days is None:
            return None

        # Apply cuts
        if features.bls_snr < self.min_snr:
            return None
        if features.bls_depth_ppm is None:
            return None
        if features.bls_depth_ppm < self.min_depth_ppm:
            return None
        if features.bls_depth_ppm > self.max_depth_ppm:
            return None
        if features.bls_n_transits < self.min_n_transits:
            return None

        # Generate deterministic candidate id
        id_source = f"{features.source_id}:{features.bls_period_days:.6f}:{features.bls_epoch_bjd:.6f}"
        candidate_id = hashlib.sha256(id_source.encode()).hexdigest()[:16]

        # Score: weighted combination of SNR, depth significance, transit count
        score = self._compute_score(features)

        # Flag potential false positives
        fp_flags = self._false_positive_checks(features)

        # Determine status
        status = CandidateStatus.NEW
        if len(fp_flags) > 0:
            status = CandidateStatus.UNVERIFIED

        return TransitCandidate(
            candidate_id=candidate_id,
            source_id=features.source_id,
            period_days=features.bls_period_days,
            epoch_bjd=features.bls_epoch_bjd,
            duration_hours=features.bls_duration_hours or 0.0,
            depth_ppm=features.bls_depth_ppm,
            snr=features.bls_snr,
            n_transits=features.bls_n_transits,
            duty_cycle=features.duty_cycle,
            even_odd_depth_ratio=features.even_odd_depth_ratio,
            status=status,
            score=score,
            false_positive_flags=fp_flags,
            notes=self._generate_notes(features, fp_flags),
            metadata={
                "ls_period_days": features.ls_period_days,
                "ls_power": features.ls_power,
                "ls_fap": features.ls_fap,
                "bls_power": features.metadata.get("bls_power"),
            },
        )

    def _compute_score(self, features: LightCurveFeatures) -> float:
        """Compute a discovery priority score (0-100).

        Higher = more interesting. Based on:
        - SNR (transit significance)
        - Transit count (more transits = more reliable)
        - Depth (deeper = easier to follow up)
        - Even/odd consistency (ratio near 1 = more planet-like)
        """
        score = 0.0

        # SNR component (0-40 points)
        snr = features.bls_snr
        if snr >= 20:
            score += 40
        elif snr >= 10:
            score += 30 + (snr - 10)
        else:
            score += snr * 3

        # Transit count component (0-20 points)
        n = features.bls_n_transits
        if n >= 10:
            score += 20
        else:
            score += n * 2

        # Depth component (0-20 points)
        depth = features.bls_depth_ppm or 0
        if depth >= 1000:
            score += 20
        else:
            score += depth / 50

        # Even/odd consistency (0-20 points)
        if features.even_odd_depth_ratio is not None:
            ratio = features.even_odd_depth_ratio
            if 0.9 <= ratio <= 1.1:
                score += 20
            elif 0.8 <= ratio <= 1.2:
                score += 10
            else:
                score += 0
        else:
            score += 10  # neutral if unknown

        return round(min(score, 100.0), 2)

    def _false_positive_checks(self, features: LightCurveFeatures) -> list[str]:
        """Run false-positive checks and return flags."""
        flags: list[str] = []

        # Even/odd depth difference (eclipsing binary indicator)
        if features.even_odd_depth_ratio is not None:
            ratio = features.even_odd_depth_ratio
            if ratio < 0.8 or ratio > 1.2:
                flags.append("even_odd_depth_mismatch")

        # Very long duration (likely stellar variability)
        if features.bls_duration_hours and features.bls_duration_hours > 12:
            flags.append("long_duration")

        # Very deep transit (likely eclipsing binary)
        if features.bls_depth_ppm and features.bls_depth_ppm > 20000:
            flags.append("very_deep_transit")

        # High duty cycle (likely not a planet)
        if features.duty_cycle > 0.3:
            flags.append("high_duty_cycle")

        # Low number of transits
        if features.bls_n_transits < 3:
            flags.append("few_transits")

        return flags

    def _generate_notes(self, features: LightCurveFeatures, flags: list[str]) -> str:
        """Generate human-readable notes."""
        parts: list[str] = []
        parts.append(f"BLS detection: P={features.bls_period_days:.4f}d, "
                     f"duration={features.bls_duration_hours:.2f}h, "
                     f"depth={features.bls_depth_ppm:.0f}ppm, "
                     f"SNR={features.bls_snr:.1f}")

        if features.ls_period_days:
            parts.append(f"LS period: {features.ls_period_days:.4f}d "
                         f"(FAP={features.ls_fap:.2e})")

        if flags:
            parts.append(f"FP flags: {', '.join(flags)}")

        return "; ".join(parts)


def generate_candidates(
    lc: LightCurve,
    min_snr: float = 5.0,
    min_depth_ppm: float = 100.0,
    max_depth_ppm: float = 50000.0,
    min_n_transits: int = 2,
    min_period_days: float = 0.1,
    max_period_days: float = 50.0,
    bls_duration_grid: list[float] | None = None,
) -> list[TransitCandidate]:
    """Convenience function: extract features and generate candidates from a light curve."""
    features = extract_features(
        lc,
        min_period_days=min_period_days,
        max_period_days=max_period_days,
        bls_duration_grid=bls_duration_grid,
    )
    generator = CandidateGenerator(
        min_snr=min_snr,
        min_depth_ppm=min_depth_ppm,
        max_depth_ppm=max_depth_ppm,
        min_n_transits=min_n_transits,
    )
    candidate = generator.generate(features)
    return [candidate] if candidate else []
