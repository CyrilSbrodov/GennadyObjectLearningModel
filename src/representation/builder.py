from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.models.schemas import ParsedHuman, TrackedHuman
from src.representation.filtering import RepresentationThresholds, apply_suppression
from src.representation.parsing_adapters import build_context, create_adapter
from src.representation.reliability import score_body_part, score_garment
from src.representation.schemas import HumanRepresentation, HumanState, Keypoint2D, MaskRegion
from src.representation.state_rules import (
    build_person_mask,
    infer_arm_state,
    infer_pose_state,
    infer_relations,
    sanitize_bbox,
)
from src.representation.utils import pose_to_keypoint_dict


@dataclass(slots=True)
class HumanRepresentationBuilder:
    """Собирает HumanRepresentation с явной поддержкой schema v1/v2."""

    min_keypoint_confidence: float = 0.2
    arm_thickness_ratio: float = 0.08
    thresholds: RepresentationThresholds = field(default_factory=RepresentationThresholds)

    def build_for_tracked_human(
        self,
        tracked_human: TrackedHuman,
        frame_index: int = 0,
        timestamp_sec: float | None = None,
        frame_shape: tuple[int, int] | None = None,
        frame: np.ndarray | None = None,
    ) -> HumanRepresentation:
        """Создает единое представление человека независимо от schema parsing."""
        human_id = f"human_{tracked_human.track_id}"
        raw_bbox = tracked_human.detection.bbox
        keypoints = pose_to_keypoint_dict(tracked_human.pose)

        parsing = tracked_human.parsed
        effective_shape = frame_shape or self._infer_shape(raw_bbox, parsing)
        bbox = sanitize_bbox(raw_bbox, effective_shape) or raw_bbox

        person_mask = self._build_person_mask(parsing)
        if person_mask is not None:
            person_mask = self._clip_region_to_bbox(person_mask, bbox=bbox, shape=effective_shape)

        body_parts = {}
        garments = {}
        if parsing is not None:
            adapter = create_adapter(parsing.schema_version)
            context = build_context(
                human_id=human_id,
                bbox=bbox,
                shape=effective_shape,
                parsing=parsing,
                keypoints=keypoints,
                frame=frame,
            )
            body_parts = adapter.build_body_parts(
                context,
                min_kp_conf=self.min_keypoint_confidence,
                arm_thickness_ratio=self.arm_thickness_ratio,
            )
            garments = adapter.build_garments(context)

        self._enrich_reliability(body_parts=body_parts, garments=garments)
        body_parts, garments = apply_suppression(body_parts=body_parts, garments=garments, thresholds=self.thresholds)
        relations = infer_relations(garments=garments, body_parts=body_parts)

        state = HumanState(
            pose_state=infer_pose_state(keypoints=keypoints, bbox=bbox),
            left_arm_state=infer_arm_state("left", keypoints=keypoints, bbox=bbox),
            right_arm_state=infer_arm_state("right", keypoints=keypoints, bbox=bbox),
        )

        dominant_garments = [
            garment.garment_type
            for garment in sorted(garments.values(), key=lambda item: item.reliability_score, reverse=True)
            if not garment.suppressed_from_overlay
        ]

        return HumanRepresentation(
            human_id=human_id,
            bbox=bbox,
            person_mask=person_mask,
            keypoints=keypoints,
            body_parts=body_parts,
            garments=garments,
            relations=relations,
            state=state,
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            identity_confidence=float(tracked_human.detection.confidence),
            parsing_confidence=float(parsing.confidence) if parsing is not None else 0.0,
            dominant_garments=list(dict.fromkeys(dominant_garments[:3])),
            has_layered_upper_clothing=any(g.garment_type == "outerwear" for g in garments.values())
            and any(g.garment_type == "upper_inner" for g in garments.values()),
            reliable_body_parts_count=sum(1 for part in body_parts.values() if part.reliability in ("high", "medium")),
            reliable_garments_count=sum(1 for garment in garments.values() if garment.reliability in ("high", "medium")),
        )

    @staticmethod
    def _build_person_mask(parsing: ParsedHuman | None) -> MaskRegion | None:
        """Строит person_mask как объединение доступных parsing-масок."""
        if parsing is None:
            return None
        return build_person_mask(parsing.masks, confidence=parsing.confidence)

    @staticmethod
    def _enrich_reliability(body_parts, garments) -> None:
        """Проставляет score и категорию reliability для сущностей."""
        for part in body_parts.values():
            score, level = score_body_part(part)
            part.reliability_score = score
            part.reliability = level
        for garment in garments.values():
            score, level = score_garment(garment)
            garment.reliability_score = score
            garment.reliability = level

    @staticmethod
    def _clip_region_to_bbox(region: MaskRegion | None, bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> MaskRegion | None:
        """Ограничивает регион bbox человека для консистентности representation."""
        if region is None:
            return None
        clean_bbox = sanitize_bbox(bbox, shape)
        if clean_bbox is None:
            return region
        x1, y1, x2, y2 = clean_bbox
        clipped = region.mask * 0
        clipped[y1:y2, x1:x2] = region.mask[y1:y2, x1:x2]
        if (clipped > 0).sum() == 0:
            return None
        return MaskRegion(mask=(clipped > 0).astype(region.mask.dtype), confidence=region.confidence)

    @staticmethod
    def _infer_shape(bbox: tuple[int, int, int, int], parsing: ParsedHuman | None) -> tuple[int, int]:
        """Определяет форму кадра/маски для безопасной валидации bbox."""
        if parsing is not None and parsing.masks:
            return next(iter(parsing.masks.values())).shape[:2]
        x1, y1, x2, y2 = bbox
        return max(y1, y2, 1), max(x1, x2, 1)
