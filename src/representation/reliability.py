from __future__ import annotations

from src.representation.schemas import BodyPart, Garment, ReliabilityLevel


def score_body_part(part: BodyPart) -> tuple[float, ReliabilityLevel]:
    """Считает численную и категориальную надежность части тела."""
    score = min(0.92, max(0.0, part.visible_fraction * 1.95))

    if "parsing" in part.evidence_sources:
        score += 0.16
    if "schema_v2" in part.evidence_sources:
        score += 0.06
    if "label:" in " ".join(part.evidence_sources):
        score += 0.04
    if "pose" in part.evidence_sources:
        score += 0.12
    if "union" in part.evidence_sources:
        score -= 0.02
    if "heuristic" in part.evidence_sources:
        score -= 0.08
    if "split_from_shoes" in part.evidence_sources:
        score -= 0.14
    if "clipped_to_bbox" in part.evidence_sources:
        score += 0.03
    if "coarse" in part.evidence_sources:
        score = min(score, 0.88)

    if "sam2-anatomy-stub" in " ".join(part.evidence_sources):
        score = min(score, 0.83)

    score = float(min(0.94, max(0.0, score)))
    return score, _score_to_level(score=score, inferred_only=part.inferred_only)


def score_garment(garment: Garment) -> tuple[float, ReliabilityLevel]:
    """Считает численную и категориальную надежность одежды."""
    score = min(0.9, max(0.0, garment.visible_fraction * 1.7))

    if "parsing" in garment.evidence_sources:
        score += 0.16
    if "schema_v2" in garment.evidence_sources:
        score += 0.04
    if "heuristic" in garment.evidence_sources:
        score -= 0.1
    if "anatomy_anchor" in garment.evidence_sources:
        score += 0.06
    if "color_contrast" in garment.evidence_sources:
        score += 0.04
    if "inner_visible_under_outer_candidate" in garment.evidence_sources:
        score += 0.06
    if "union" in garment.evidence_sources:
        score -= 0.02
    if "clipped_to_bbox" in garment.evidence_sources:
        score += 0.03

    if garment.inferred_only:
        score = min(score, 0.82)
    if garment.garment_type == "outerwear":
        score = min(score, 0.8)

    score = float(min(0.9, max(0.0, score)))
    return score, _score_to_level(score=score, inferred_only=garment.inferred_only)


def _score_to_level(score: float, inferred_only: bool) -> ReliabilityLevel:
    """Переводит score в дискретный уровень надежности."""
    if inferred_only and score < 0.48:
        return "inferred"
    if score >= 0.76:
        return "high"
    if score >= 0.5:
        return "medium"
    if score >= 0.26:
        return "low"
    return "inferred"
