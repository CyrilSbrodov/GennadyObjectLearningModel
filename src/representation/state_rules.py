from __future__ import annotations

import numpy as np

from src.representation.schemas import BodyPart, Garment, HumanPoseState, Keypoint2D, LimbState, MaskRegion, RelationEdge

# Порог близости бедра/колена по вертикали для сидения.
SITTING_KNEE_DELTA_RATIO = 0.1
# Порог почти горизонтального тела для состояния lying.
LYING_SHOULDER_HIP_VERTICAL_RATIO = 0.07
# Порог вертикального положения кисти относительно плеча для raised.
ARM_RAISED_Y_DELTA_RATIO = 0.05
# Порог схожести высоты плеча и кисти для bent.
ARM_BENT_Y_DELTA_RATIO = 0.08


def infer_pose_state(keypoints: dict[str, Keypoint2D], bbox: tuple[int, int, int, int]) -> HumanPoseState:
    """Оценивает положение тела: standing/sitting/lying/unknown по простым правилам."""
    x1, y1, x2, y2 = bbox
    height = max(1.0, float(y2 - y1))
    left_shoulder = keypoints.get("left_shoulder")
    right_shoulder = keypoints.get("right_shoulder")
    left_hip = keypoints.get("left_hip")
    right_hip = keypoints.get("right_hip")
    left_knee = keypoints.get("left_knee")
    right_knee = keypoints.get("right_knee")

    if left_shoulder and right_shoulder and left_hip and right_hip:
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2.0
        hip_y = (left_hip.y + right_hip.y) / 2.0
        if abs(hip_y - shoulder_y) < height * LYING_SHOULDER_HIP_VERTICAL_RATIO:
            return "lying"

    if left_hip and right_hip and left_knee and right_knee:
        hip_y = (left_hip.y + right_hip.y) / 2.0
        knee_y = (left_knee.y + right_knee.y) / 2.0
        if abs(knee_y - hip_y) < height * SITTING_KNEE_DELTA_RATIO:
            return "sitting"
        return "standing"

    return "unknown_pose"


def infer_arm_state(side: str, keypoints: dict[str, Keypoint2D], bbox: tuple[int, int, int, int]) -> LimbState:
    """Оценивает состояние руки raised/lowered/bent/extended/unknown."""
    x1, y1, x2, y2 = bbox
    height = max(1.0, float(y2 - y1))
    shoulder = keypoints.get(f"{side}_shoulder")
    elbow = keypoints.get(f"{side}_elbow")
    wrist = keypoints.get(f"{side}_wrist")

    if shoulder is None or wrist is None:
        return "unknown_limb_state"
    if wrist.y < shoulder.y - height * ARM_RAISED_Y_DELTA_RATIO:
        return "raised"
    if elbow is not None and abs(wrist.y - shoulder.y) < height * ARM_BENT_Y_DELTA_RATIO:
        return "bent"
    if elbow is not None and wrist.y > elbow.y > shoulder.y:
        return "lowered"
    return "extended"


def estimate_visible_fraction(region: MaskRegion | None, bbox: tuple[int, int, int, int]) -> float:
    """Оценивает долю видимости по площади маски относительно bbox."""
    if region is None:
        return 0.0
    clean_bbox = sanitize_bbox(bbox, region.mask.shape[:2])
    if clean_bbox is None:
        return 0.0
    x1, y1, x2, y2 = clean_bbox
    bbox_area = max(1, (x2 - x1) * (y2 - y1))
    mask_crop = region.mask[y1:y2, x1:x2]
    mask_area = int(np.count_nonzero(mask_crop > 0))
    return float(np.clip(mask_area / bbox_area, 0.0, 1.0))


def build_person_mask(masks: dict[str, np.ndarray], confidence: float) -> MaskRegion | None:
    """Строит грубую person_mask как объединение масок парсинга."""
    if not masks:
        return None
    union_mask: np.ndarray | None = None
    for mask in masks.values():
        current = (mask > 0).astype(np.uint8)
        union_mask = current if union_mask is None else np.maximum(union_mask, current)
    if union_mask is None:
        return None
    return MaskRegion(mask=union_mask.astype(np.uint8), confidence=float(confidence))


def sanitize_bbox(bbox: tuple[int, int, int, int], shape_hw: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """Нормализует и обрезает bbox в границах массива."""
    height, width = shape_hw
    if height <= 0 or width <= 0:
        return None
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def infer_relations(garments: dict[str, Garment], body_parts: dict[str, BodyPart]) -> list[RelationEdge]:
    """Создает набор отношений с учетом надежности и подавления сущностей."""
    relations: list[RelationEdge] = []
    upper_id: str | None = None
    outer_id: str | None = None

    for garment_id, garment in garments.items():
        if not _has_pixels(garment.region) or garment.suppressed_from_relations:
            continue
        if garment.reliability_score < 0.33:
            continue

        garment_type = garment.garment_type
        if garment_type == "upper_inner":
            upper_id = garment_id
        if garment_type == "outerwear" and garment.is_outer_layer_candidate:
            outer_id = garment_id

        for part_id in garment.attached_body_parts:
            part = body_parts.get(part_id)
            if part is None or not _has_pixels(part.region):
                continue
            if part.suppressed_from_relations or part.reliability_score < 0.33:
                continue

            confidence = min(
                0.98,
                _attachment_confidence(part_id) * (0.65 + 0.35 * garment.reliability_score) * (0.65 + 0.35 * part.reliability_score),
            )
            relations.append(RelationEdge(garment_id, part_id, "attached_to", float(confidence)))

    if upper_id and outer_id:
        upper = garments.get(upper_id)
        outer = garments.get(outer_id)
        if upper is not None and outer is not None:
            if (
                "inner_visible_under_outer_candidate" in upper.evidence_sources
                and outer.reliability_score >= 0.42
                and upper.reliability_score >= 0.36
            ):
                covers_confidence = 0.55 + 0.25 * min(outer.reliability_score, upper.reliability_score)
                relations.append(
                    RelationEdge(source_id=outer_id, target_id=upper_id, relation_type="covers", confidence=float(min(0.92, covers_confidence)))
                )
    return relations


def _has_pixels(region: MaskRegion | None) -> bool:
    """Проверяет, что регион существует и содержит ненулевые пиксели."""
    if region is None:
        return False
    return bool(np.count_nonzero(region.mask > 0))


def _attachment_confidence(part_id: str) -> float:
    """Возвращает базовую уверенность связи attached_to для части тела."""
    if "foot" in part_id:
        return 0.79
    if "thigh" in part_id or "pelvis" in part_id:
        return 0.81
    if "upper_arm" in part_id or "forearm" in part_id:
        return 0.76
    if "arm" in part_id:
        return 0.74
    return 0.82
