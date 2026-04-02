from __future__ import annotations

from dataclasses import dataclass

from src.representation.schemas import BodyPart, Garment


@dataclass(slots=True)
class RepresentationThresholds:
    """Пороговые значения для подавления слабых сущностей."""

    min_visible_fraction_for_reliable_part: float = 0.02
    min_visible_fraction_for_reliable_garment: float = 0.02
    min_visible_fraction_for_overlay: float = 0.025
    min_visible_fraction_for_relations: float = 0.03
    min_reliability_score_for_relations: float = 0.33


def apply_suppression(
    body_parts: dict[str, BodyPart],
    garments: dict[str, Garment],
    thresholds: RepresentationThresholds,
) -> tuple[dict[str, BodyPart], dict[str, Garment]]:
    """Маркирует слабые сущности и удаляет откровенный шум."""
    filtered_parts: dict[str, BodyPart] = {}
    for part_id, part in body_parts.items():
        part.suppressed_from_overlay = part.visible_fraction < thresholds.min_visible_fraction_for_overlay
        part.suppressed_from_relations = (
            part.visible_fraction < thresholds.min_visible_fraction_for_relations
            or part.reliability_score < thresholds.min_reliability_score_for_relations
        )

        if part.visible_fraction < 0.005 and part.reliability_score < 0.2:
            continue
        filtered_parts[part_id] = part

    filtered_garments: dict[str, Garment] = {}
    for garment_id, garment in garments.items():
        garment.suppressed_from_overlay = garment.visible_fraction < thresholds.min_visible_fraction_for_overlay
        garment.suppressed_from_relations = (
            garment.visible_fraction < thresholds.min_visible_fraction_for_relations
            or garment.reliability_score < thresholds.min_reliability_score_for_relations
        )

        if garment.visible_fraction < 0.006 and garment.reliability_score < 0.2:
            continue
        filtered_garments[garment_id] = garment

    return filtered_parts, filtered_garments
