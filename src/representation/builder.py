from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from src.models.schemas import ParsedHuman, TrackedHuman
from src.representation.filtering import RepresentationThresholds, apply_suppression
from src.representation.garment_reasoning import build_upper_garments, infer_upper_garment_candidates
from src.representation.reliability import score_body_part, score_garment
from src.representation.schemas import BodyPart, Garment, HumanRepresentation, HumanState, Keypoint2D, MaskRegion
from src.representation.state_rules import (
    build_person_mask,
    estimate_visible_fraction,
    infer_arm_state,
    infer_pose_state,
    infer_relations,
    sanitize_bbox,
)
from src.representation.utils import pose_to_keypoint_dict


@dataclass(slots=True)
class HumanRepresentationBuilder:
    """Собирает HumanRepresentation v1 из трека без нейросетевых зависимостей."""

    # Минимальная уверенность ключевой точки, чтобы использовать ее в маске руки.
    min_keypoint_confidence: float = 0.2
    # Толщина линии сегмента руки относительно размера bbox.
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
        """Создает представление человека устойчиво к неполным данным."""
        human_id = f"human_{tracked_human.track_id}"
        raw_bbox = tracked_human.detection.bbox
        keypoints = pose_to_keypoint_dict(tracked_human.pose)

        parsing = tracked_human.parsed
        effective_shape = frame_shape or self._infer_shape(raw_bbox, parsing)
        bbox = sanitize_bbox(raw_bbox, effective_shape) or raw_bbox

        person_mask = self._clip_region_to_bbox(self._build_person_mask(parsing), bbox=bbox, shape=effective_shape)
        body_parts = self._build_body_parts(bbox=bbox, keypoints=keypoints, parsing=parsing, shape=effective_shape)
        garments = self._build_garments(human_id=human_id, bbox=bbox, parsing=parsing, shape=effective_shape, frame=frame)

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

    def _build_person_mask(self, parsing: ParsedHuman | None) -> MaskRegion | None:
        """Строит person_mask как объединение всех доступных parsing-масок."""
        if parsing is None:
            return None
        return build_person_mask(parsing.masks, confidence=parsing.confidence)

    def _build_body_parts(
        self,
        bbox: tuple[int, int, int, int],
        keypoints: dict[str, Keypoint2D],
        parsing: ParsedHuman | None,
        shape: tuple[int, int],
    ) -> dict[str, BodyPart]:
        """Собирает набор body parts из парсинга и безопасных эвристик."""
        masks = parsing.masks if parsing is not None else {}
        parts: dict[str, BodyPart] = {}

        face_region = self._safe_region(masks.get("face"), parsing)
        hair_region = self._safe_region(masks.get("hair"), parsing)
        head_region = self._union_regions([face_region, hair_region])

        region_by_part: dict[str, tuple[MaskRegion | None, list[str], bool]] = {
            "head": (self._clip_region_to_bbox(head_region, bbox=bbox, shape=shape), ["parsing", "union"], False),
            "face": (self._clip_region_to_bbox(face_region, bbox=bbox, shape=shape), ["parsing"], False),
            "hair": (self._clip_region_to_bbox(hair_region, bbox=bbox, shape=shape), ["parsing"], False),
            "torso": (
                self._clip_region_to_bbox(self._safe_region(masks.get("upper_clothes"), parsing), bbox=bbox, shape=shape),
                ["parsing"],
                False,
            ),
            "left_hand": (
                self._clip_region_to_bbox(self._safe_region(masks.get("left_hand"), parsing), bbox=bbox, shape=shape),
                ["parsing"],
                False,
            ),
            "right_hand": (
                self._clip_region_to_bbox(self._safe_region(masks.get("right_hand"), parsing), bbox=bbox, shape=shape),
                ["parsing"],
                False,
            ),
            "left_leg": (
                self._clip_region_to_bbox(self._safe_region(masks.get("left_leg"), parsing), bbox=bbox, shape=shape),
                ["parsing"],
                False,
            ),
            "right_leg": (
                self._clip_region_to_bbox(self._safe_region(masks.get("right_leg"), parsing), bbox=bbox, shape=shape),
                ["parsing"],
                False,
            ),
        }

        arm_regions = self._build_arm_regions(keypoints=keypoints, bbox=bbox, masks=masks, shape=shape)
        if arm_regions["left_arm"] is not None:
            region_by_part["left_arm"] = (
                self._clip_region_to_bbox(arm_regions["left_arm"], bbox=bbox, shape=shape),
                ["pose", "heuristic"],
                True,
            )
        if arm_regions["right_arm"] is not None:
            region_by_part["right_arm"] = (
                self._clip_region_to_bbox(arm_regions["right_arm"], bbox=bbox, shape=shape),
                ["pose", "heuristic"],
                True,
            )

        shoes_mask = masks.get("shoes")
        left_foot_region, right_foot_region = self._split_shoes_region(shoes_mask=shoes_mask, parsing=parsing, bbox=bbox)
        if left_foot_region is not None:
            region_by_part["left_foot"] = (
                self._clip_region_to_bbox(left_foot_region, bbox=bbox, shape=shape),
                ["parsing", "split_from_shoes"],
                True,
            )
        if right_foot_region is not None:
            region_by_part["right_foot"] = (
                self._clip_region_to_bbox(right_foot_region, bbox=bbox, shape=shape),
                ["parsing", "split_from_shoes"],
                True,
            )

        if "left_shoulder" in keypoints and "right_shoulder" in keypoints:
            neck_region = self._build_neck_region(bbox, parsing)
            if neck_region is not None:
                region_by_part["neck"] = (self._clip_region_to_bbox(neck_region, bbox=bbox, shape=shape), ["heuristic", "pose"], True)

        for part_name, (region, evidence_sources, inferred_only) in region_by_part.items():
            visible_fraction = estimate_visible_fraction(region=region, bbox=bbox)
            if region is not None:
                evidence_sources = [*evidence_sources, "clipped_to_bbox"]
            parts[part_name] = BodyPart(
                part_id=part_name,
                name=part_name,
                region=region,
                visible_fraction=visible_fraction,
                occluded=(region is None) or visible_fraction < 0.1,
                evidence_sources=evidence_sources,
                inferred_only=inferred_only,
            )
        return parts

    def _build_garments(
        self,
        human_id: str,
        bbox: tuple[int, int, int, int],
        parsing: ParsedHuman | None,
        shape: tuple[int, int],
        frame: np.ndarray | None,
    ) -> dict[str, Garment]:
        """Строит одежду с эвристическим разбором верхних слоев."""
        if parsing is None:
            return {}
        garments: dict[str, Garment] = {}

        upper_hypothesis = infer_upper_garment_candidates(frame=frame, bbox=bbox, parsing=parsing)
        upper_garments = build_upper_garments(human_id=human_id, bbox=bbox, hypothesis=upper_hypothesis)
        for garment in upper_garments.values():
            if garment.region is not None:
                garment.region = self._clip_region_to_bbox(garment.region, bbox=bbox, shape=shape)
                garment.visible_fraction = estimate_visible_fraction(garment.region, bbox)
                if garment.region is not None:
                    garment.evidence_sources.append("clipped_to_bbox")
        garments.update(upper_garments)

        if "lower_clothes" in parsing.masks:
            region = self._clip_region_to_bbox(self._safe_region(parsing.masks["lower_clothes"], parsing), bbox=bbox, shape=shape)
            garments[f"{human_id}_garment_lower_0"] = Garment(
                garment_id=f"{human_id}_garment_lower_0",
                garment_type="pants",
                region=region,
                visible_fraction=estimate_visible_fraction(region, bbox),
                state="worn",
                attached_body_parts=["left_leg", "right_leg"],
                evidence_sources=["parsing", "clipped_to_bbox"],
            )

        if "shoes" in parsing.masks:
            region = self._clip_region_to_bbox(self._safe_region(parsing.masks["shoes"], parsing), bbox=bbox, shape=shape)
            garments[f"{human_id}_garment_shoes_0"] = Garment(
                garment_id=f"{human_id}_garment_shoes_0",
                garment_type="shoes",
                region=region,
                visible_fraction=estimate_visible_fraction(region, bbox),
                state="worn",
                attached_body_parts=["left_foot", "right_foot"],
                evidence_sources=["parsing", "clipped_to_bbox"],
            )

        if not garments and "upper_clothes" in parsing.masks:
            region = self._clip_region_to_bbox(self._safe_region(parsing.masks["upper_clothes"], parsing), bbox=bbox, shape=shape)
            garments[f"{human_id}_garment_upper_0"] = Garment(
                garment_id=f"{human_id}_garment_upper_0",
                garment_type="upper_inner",
                region=region,
                visible_fraction=estimate_visible_fraction(region, bbox),
                state="worn",
                attached_body_parts=["torso", "left_arm", "right_arm"],
                evidence_sources=["parsing", "clipped_to_bbox"],
            )

        return garments

    def _enrich_reliability(self, body_parts: dict[str, BodyPart], garments: dict[str, Garment]) -> None:
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
    def _safe_region(mask: np.ndarray | None, parsing: ParsedHuman | None) -> MaskRegion | None:
        """Преобразует mask в MaskRegion, если маска существует."""
        if mask is None or parsing is None:
            return None
        return MaskRegion(mask=(mask > 0).astype(np.uint8), confidence=float(parsing.confidence))

    def _build_arm_regions(
        self,
        keypoints: dict[str, Keypoint2D],
        bbox: tuple[int, int, int, int],
        masks: dict[str, np.ndarray],
        shape: tuple[int, int],
    ) -> dict[str, MaskRegion | None]:
        """Строит регионы рук по линиям плечо-локоть-запястье; без данных возвращает None."""
        mask_shape = next(iter(masks.values())).shape[:2] if masks else shape
        left_arm = self._draw_arm_mask("left", keypoints, bbox, mask_shape)
        right_arm = self._draw_arm_mask("right", keypoints, bbox, mask_shape)
        return {"left_arm": left_arm, "right_arm": right_arm}

    def _draw_arm_mask(
        self,
        side: str,
        keypoints: dict[str, Keypoint2D],
        bbox: tuple[int, int, int, int],
        shape: tuple[int, int],
    ) -> MaskRegion | None:
        """Рисует маску руки по сегментам ключевых точек, если они доступны и надежны."""
        shoulder = keypoints.get(f"{side}_shoulder")
        elbow = keypoints.get(f"{side}_elbow")
        wrist = keypoints.get(f"{side}_wrist")
        if shoulder is None or wrist is None:
            return None
        if shoulder.confidence < self.min_keypoint_confidence or wrist.confidence < self.min_keypoint_confidence:
            return None

        height = max(1, bbox[3] - bbox[1])
        thickness = max(2, int(height * self.arm_thickness_ratio))
        arm_mask = np.zeros(shape, dtype=np.uint8)
        pts: list[tuple[int, int]] = [(int(shoulder.x), int(shoulder.y))]
        if elbow is not None and elbow.confidence >= self.min_keypoint_confidence:
            pts.append((int(elbow.x), int(elbow.y)))
        pts.append((int(wrist.x), int(wrist.y)))

        for i in range(len(pts) - 1):
            cv2.line(arm_mask, pts[i], pts[i + 1], 1, thickness=thickness)

        if np.count_nonzero(arm_mask) == 0:
            return None
        return MaskRegion(mask=arm_mask, confidence=0.6)

    def _split_shoes_region(
        self,
        shoes_mask: np.ndarray | None,
        parsing: ParsedHuman | None,
        bbox: tuple[int, int, int, int],
    ) -> tuple[MaskRegion | None, MaskRegion | None]:
        """Делит общую shoes-маску на левую/правую часть относительно центра bbox."""
        if shoes_mask is None or parsing is None:
            return None, None
        mask = (shoes_mask > 0).astype(np.uint8)
        if np.count_nonzero(mask) == 0:
            return None, None

        clean_bbox = sanitize_bbox(bbox, mask.shape[:2])
        if clean_bbox is None:
            return None, None
        x1, _, x2, _ = clean_bbox
        center_x = x1 + max(1, (x2 - x1) // 2)
        left = np.zeros_like(mask)
        right = np.zeros_like(mask)
        left[:, :center_x] = mask[:, :center_x]
        right[:, center_x:] = mask[:, center_x:]

        left_region = MaskRegion(mask=left, confidence=float(parsing.confidence)) if np.count_nonzero(left) > 0 else None
        right_region = MaskRegion(mask=right, confidence=float(parsing.confidence)) if np.count_nonzero(right) > 0 else None
        return left_region, right_region

    def _union_regions(self, regions: list[MaskRegion | None]) -> MaskRegion | None:
        """Объединяет несколько регионов в один общий регион."""
        valid_regions = [region for region in regions if region is not None]
        if not valid_regions:
            return None
        union_mask = valid_regions[0].mask.copy().astype(np.uint8)
        confidence = valid_regions[0].confidence
        for region in valid_regions[1:]:
            union_mask = np.maximum(union_mask, (region.mask > 0).astype(np.uint8))
            confidence = max(confidence, region.confidence)
        if np.count_nonzero(union_mask) == 0:
            return None
        return MaskRegion(mask=union_mask, confidence=float(confidence))

    @staticmethod
    def _clip_region_to_bbox(
        region: MaskRegion | None,
        bbox: tuple[int, int, int, int],
        shape: tuple[int, int],
    ) -> MaskRegion | None:
        """Ограничивает регион bbox человека для повышения консистентности representation."""
        if region is None:
            return None
        clean_bbox = sanitize_bbox(bbox, shape)
        if clean_bbox is None:
            return region
        x1, y1, x2, y2 = clean_bbox
        clipped = np.zeros_like(region.mask, dtype=np.uint8)
        clipped[y1:y2, x1:x2] = (region.mask[y1:y2, x1:x2] > 0).astype(np.uint8)
        if np.count_nonzero(clipped) == 0:
            return None
        return MaskRegion(mask=clipped, confidence=region.confidence)

    @staticmethod
    def _build_neck_region(bbox: tuple[int, int, int, int], parsing: ParsedHuman | None) -> MaskRegion | None:
        """Создает узкую эвристическую область шеи внутри bbox."""
        if parsing is None:
            return None
        x1, y1, x2, y2 = bbox
        h, w = next(iter(parsing.masks.values())).shape[:2] if parsing.masks else (y2 + 1, x2 + 1)
        clean_bbox = sanitize_bbox((x1, y1, x2, y2), (h, w))
        if clean_bbox is None:
            return None
        x1, y1, x2, y2 = clean_bbox
        mask = np.zeros((h, w), dtype=np.uint8)
        neck_w = max(2, (x2 - x1) // 6)
        neck_h = max(2, (y2 - y1) // 12)
        cx = (x1 + x2) // 2
        y_top = y1 + max(1, (y2 - y1) // 8)
        x_start = max(0, cx - neck_w // 2)
        x_end = min(w, x_start + neck_w)
        y_end = min(h, y_top + neck_h)
        mask[max(0, y_top):y_end, x_start:x_end] = 1
        return MaskRegion(mask=mask, confidence=float(parsing.confidence) * 0.8)

    @staticmethod
    def _infer_shape(bbox: tuple[int, int, int, int], parsing: ParsedHuman | None) -> tuple[int, int]:
        """Определяет форму кадра/маски для безопасной валидации bbox."""
        if parsing is not None and parsing.masks:
            return next(iter(parsing.masks.values())).shape[:2]
        x1, y1, x2, y2 = bbox
        return max(y1, y2, 1), max(x1, x2, 1)
