from __future__ import annotations

from src.representation.schemas import BodyPart, Garment, ReliabilityLevel


def score_body_part(part: BodyPart) -> tuple[float, ReliabilityLevel]:
    """Считает численную и категориальную надежность части тела."""
    score = min(1.0, max(0.0, part.visible_fraction * 2.2))

    if "parsing" in part.evidence_sources:
        score += 0.2
    if "pose" in part.evidence_sources:
        score += 0.2
    if "union" in part.evidence_sources:
        score += 0.08
    if "heuristic" in part.evidence_sources:
        score -= 0.05
    if "split_from_shoes" in part.evidence_sources:
        score -= 0.12
    if "clipped_to_bbox" in part.evidence_sources:
        score += 0.05

    score = float(min(1.0, max(0.0, score)))
    return score, _score_to_level(score=score, inferred_only=part.inferred_only)


def score_garment(garment: Garment) -> tuple[float, ReliabilityLevel]:
    """Считает численную и категориальную надежность одежды."""
    score = min(1.0, max(0.0, garment.visible_fraction * 2.0))

    if "parsing" in garment.evidence_sources:
        score += 0.25
    if "heuristic" in garment.evidence_sources:
        score -= 0.06
    if "color_contrast" in garment.evidence_sources:
        score += 0.06
    if "inner_visible_under_outer_candidate" in garment.evidence_sources:
        score += 0.08
    if "clipped_to_bbox" in garment.evidence_sources:
        score += 0.05

    if garment.garment_type == "outerwear" and "parsing" not in garment.evidence_sources:
        score = min(score, 0.74)

    score = float(min(1.0, max(0.0, score)))
    return score, _score_to_level(score=score, inferred_only=garment.inferred_only)


def _score_to_level(score: float, inferred_only: bool) -> ReliabilityLevel:
    """Переводит score в дискретный уровень надежности."""
    if inferred_only and score < 0.5:
        return "inferred"
    if score >= 0.78:
        return "high"
    if score >= 0.52:
        return "medium"
    if score >= 0.26:
        return "low"
    return "inferred"
