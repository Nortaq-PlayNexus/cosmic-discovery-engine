"""Discovery pipeline: ingestion, feature extraction, candidate generation, validation."""

from .features import LightCurveFeatures, extract_features
from .candidates import generate_candidates, CandidateGenerator
from .crossmatch import CrossMatchResult, crossmatch_candidate

__all__ = [
    "LightCurveFeatures",
    "extract_features",
    "generate_candidates",
    "CandidateGenerator",
    "CrossMatchResult",
    "crossmatch_candidate",
]
