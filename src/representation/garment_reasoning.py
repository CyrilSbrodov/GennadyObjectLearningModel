from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.models.schemas import ParsedHuman
from src.representation.schemas import MaskRegion
from src.representation.state_rules import sanitize_bbox


@dataclass(slots=True)
class UpperGarmentHypothesis:
    """Гипотеза о слоях верхней одежды."""

    upper_inner: MaskRegion | None
    outerwear: MaskRegion | None
    has_layered_upper: bool
    evidence: list[str]


def infer_upper_garment_candidates_v1(
    frame: np.ndarray | None,
    bbox: tuple[int, int, int, int],
    parsing: ParsedHuman,
) -> UpperGarmentHypothesis:
    """Строит консервативную гипотезу слоёв верхней одежды для schema v1."""
    upper_mask = parsing.masks.get("upper_clothes")
    if upper_mask is None:
        return UpperGarmentHypothesis(None, None, False, [])

    upper_region = MaskRegion(mask=(upper_mask > 0).astype(np.uint8), confidence=float(parsing.label_confidence.get("upper_clothes", parsing.confidence)))
    clean_bbox = sanitize_bbox(bbox, upper_region.mask.shape[:2])
    if clean_bbox is None:
        return UpperGarmentHypothesis(upper_region, None, False, ["parsing", "schema_v1"])

    inner_region = _detect_inner_visible_core(frame=frame, base_region=upper_region, bbox=clean_bbox)
    outer_region = _detect_outerwear_candidate(base_region=upper_region, inner_region=inner_region, bbox=clean_bbox)
    evidence = ["schema_v1", "parsing"]
    if inner_region is not None:
        evidence.append("inner_visible_under_outer_candidate")
    if outer_region is not None:
        evidence.extend(["heuristic", "color_contrast"])

    has_layered = inner_region is not None and outer_region is not None
    if not has_layered:
        return UpperGarmentHypothesis(upper_inner=upper_region, outerwear=None, has_layered_upper=False, evidence=evidence)
    return UpperGarmentHypothesis(upper_inner=inner_region, outerwear=outer_region, has_layered_upper=True, evidence=evidence)


def infer_upper_garment_candidates_v2(
    frame: np.ndarray | None,
    bbox: tuple[int, int, int, int],
    parsing: ParsedHuman,
) -> UpperGarmentHypothesis:
    """Строит layered-гипотезу верхней одежды на анатомических якорях schema v2."""
    clean_bbox = sanitize_bbox(bbox, next(iter(parsing.masks.values())).shape[:2]) if parsing.masks else None
    if clean_bbox is None:
        return UpperGarmentHypothesis(None, None, False, ["schema_v2"])

    torso_core = _union_masks(
        parsing,
        ["chest_left", "chest_right", "abdomen", "pelvis"],
    )
    if torso_core is None:
        return UpperGarmentHypothesis(None, None, False, ["schema_v2", "missing_torso_core"])

    arm_envelope = _union_masks(
        parsing,
        ["shoulder_left", "upper_arm_left", "forearm_left", "hand_left", "shoulder_right", "upper_arm_right", "forearm_right", "hand_right"],
    )

    base_mask = torso_core.mask.copy().astype(np.uint8)
    if arm_envelope is not None:
        base_mask = np.maximum(base_mask, (arm_envelope.mask > 0).astype(np.uint8))
    base_region = MaskRegion(mask=base_mask, confidence=float(np.mean([torso_core.confidence, arm_envelope.confidence if arm_envelope else torso_core.confidence])))

    corridor = _central_torso_corridor(mask_shape=base_mask.shape, bbox=clean_bbox)
    inner_mask = ((torso_core.mask > 0) & (corridor > 0)).astype(np.uint8)
    inner_region = None
    if np.count_nonzero(inner_mask) > 18:
        inner_conf = torso_core.confidence * 0.84
        if frame is not None:
            contrast = _mean_contrast(frame, outer=(base_mask > 0), inner=(inner_mask > 0))
            if contrast > 6.5:
                inner_region = MaskRegion(mask=inner_mask, confidence=float(min(0.9, inner_conf + 0.03)))
            else:
                inner_region = MaskRegion(mask=inner_mask, confidence=float(inner_conf * 0.92))
        else:
            inner_region = MaskRegion(mask=inner_mask, confidence=float(inner_conf))

    outer_region = _detect_outerwear_candidate(base_region=base_region, inner_region=inner_region, bbox=clean_bbox)

    evidence = ["schema_v2", "parsing", "anatomy_anchor"]
    if arm_envelope is not None:
        evidence.append("arm_envelope")
    if inner_region is not None:
        evidence.append("inner_visible_under_outer_candidate")
    if outer_region is not None:
        evidence.extend(["heuristic", "color_contrast"])

    if inner_region is None:
        # Без внутреннего ядра возвращаем цельный верх.
        return UpperGarmentHypothesis(upper_inner=base_region, outerwear=None, has_layered_upper=False, evidence=evidence)

    if outer_region is None:
        return UpperGarmentHypothesis(upper_inner=inner_region, outerwear=None, has_layered_upper=False, evidence=evidence)

    return UpperGarmentHypothesis(upper_inner=inner_region, outerwear=outer_region, has_layered_upper=True, evidence=evidence)


def _detect_inner_visible_core(
    frame: np.ndarray | None,
    base_region: MaskRegion,
    bbox: tuple[int, int, int, int],
) -> MaskRegion | None:
    """Выделяет внутреннюю центральную область верхней одежды."""
    core = _central_torso_corridor(mask_shape=base_region.mask.shape, bbox=bbox)
    inner_mask = ((base_region.mask > 0) & (core > 0)).astype(np.uint8)
    if np.count_nonzero(inner_mask) < 18:
        return None

    if frame is not None:
        contrast = _mean_contrast(frame, outer=(base_region.mask > 0), inner=(inner_mask > 0))
        if contrast < 7.0:
            return None

    return MaskRegion(mask=inner_mask, confidence=base_region.confidence * 0.82)


def _detect_outerwear_candidate(
    base_region: MaskRegion,
    inner_region: MaskRegion | None,
    bbox: tuple[int, int, int, int],
) -> MaskRegion | None:
    """Выделяет внешний слой как область вокруг внутреннего ядра."""
    if inner_region is None:
        return None

    x1, y1, x2, y2 = bbox
    base = (base_region.mask > 0).astype(np.uint8)
    inner = (inner_region.mask > 0).astype(np.uint8)
    outer_mask = ((base == 1) & (inner == 0)).astype(np.uint8)
    if np.count_nonzero(outer_mask) < 30:
        return None

    bbox_area = max(1, (x2 - x1) * (y2 - y1))
    outer_fraction = np.count_nonzero(outer_mask[y1:y2, x1:x2]) / bbox_area
    inner_fraction = np.count_nonzero(inner[y1:y2, x1:x2]) / bbox_area
    if outer_fraction < 0.04 or inner_fraction < 0.01:
        return None

    conf = max(inner_region.confidence * 0.86, base_region.confidence * 0.74)
    return MaskRegion(mask=outer_mask, confidence=float(conf))


def _union_masks(parsing: ParsedHuman, labels: list[str]) -> MaskRegion | None:
    """Объединяет набор масок и усредняет confidence по лейблам."""
    masks: list[np.ndarray] = []
    confidences: list[float] = []
    for label in labels:
        mask = parsing.masks.get(label)
        if mask is None:
            continue
        bin_mask = (mask > 0).astype(np.uint8)
        if np.count_nonzero(bin_mask) == 0:
            continue
        masks.append(bin_mask)
        confidences.append(float(parsing.label_confidence.get(label, parsing.confidence)))

    if not masks:
        return None
    out = masks[0].copy()
    for m in masks[1:]:
        out = np.maximum(out, m)
    return MaskRegion(mask=out, confidence=float(np.mean(confidences)))


def _central_torso_corridor(mask_shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Строит центральный коридор торса внутри bbox."""
    x1, y1, x2, y2 = bbox
    h = max(1, y2 - y1)
    w = max(1, x2 - x1)
    center_w = max(3, int(w * 0.34))
    cx = (x1 + x2) // 2
    core_x1 = max(x1, cx - center_w // 2)
    core_x2 = min(x2, core_x1 + center_w)
    core_y1 = y1 + int(h * 0.18)
    core_y2 = y1 + int(h * 0.78)
    core = np.zeros(mask_shape, dtype=np.uint8)
    core[core_y1:core_y2, core_x1:core_x2] = 1
    return core


def _mean_contrast(frame: np.ndarray, outer: np.ndarray, inner: np.ndarray) -> float:
    """Считает контраст внутренней области относительно внешней."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    outer_pixels = gray[outer]
    inner_pixels = gray[inner]
    if outer_pixels.size == 0 or inner_pixels.size == 0:
        return 0.0
    return float(np.mean(inner_pixels) - np.mean(outer_pixels))
