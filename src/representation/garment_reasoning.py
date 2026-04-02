from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.models.schemas import ParsedHuman
from src.representation.schemas import Garment, MaskRegion
from src.representation.state_rules import estimate_visible_fraction, sanitize_bbox


@dataclass(slots=True)
class UpperGarmentHypothesis:
    """Гипотеза о слоях верхней одежды."""

    upper_inner: MaskRegion | None
    outerwear: MaskRegion | None
    has_layered_upper: bool
    evidence: list[str]


def infer_upper_garment_candidates(
    frame: np.ndarray | None,
    bbox: tuple[int, int, int, int],
    parsing: ParsedHuman,
) -> UpperGarmentHypothesis:
    """Строит консервативную гипотезу слоёв верхней одежды."""
    upper_mask = parsing.masks.get("upper_clothes")
    if upper_mask is None:
        return UpperGarmentHypothesis(None, None, False, [])

    upper_region = MaskRegion(mask=(upper_mask > 0).astype(np.uint8), confidence=float(parsing.confidence))
    clean_bbox = sanitize_bbox(bbox, upper_region.mask.shape[:2])
    if clean_bbox is None:
        return UpperGarmentHypothesis(upper_region, None, False, ["parsing"])

    inner_region = detect_inner_visible_core(frame=frame, upper_region=upper_region, bbox=clean_bbox)
    outer_region = detect_outerwear_candidate(upper_region=upper_region, inner_region=inner_region, bbox=clean_bbox)

    evidence = ["parsing"]
    if inner_region is not None:
        evidence.append("inner_visible_under_outer_candidate")
    if outer_region is not None:
        evidence.extend(["heuristic", "color_contrast"])

    has_layered = inner_region is not None and outer_region is not None
    if not has_layered:
        return UpperGarmentHypothesis(upper_inner=upper_region, outerwear=None, has_layered_upper=False, evidence=evidence)

    return UpperGarmentHypothesis(upper_inner=inner_region, outerwear=outer_region, has_layered_upper=True, evidence=evidence)


def detect_inner_visible_core(
    frame: np.ndarray | None,
    upper_region: MaskRegion,
    bbox: tuple[int, int, int, int],
) -> MaskRegion | None:
    """Выделяет внутреннюю центральную область верхней одежды."""
    x1, y1, x2, y2 = bbox
    h = max(1, y2 - y1)
    w = max(1, x2 - x1)

    center_w = max(2, int(w * 0.32))
    cx = (x1 + x2) // 2
    core_x1 = max(x1, cx - center_w // 2)
    core_x2 = min(x2, core_x1 + center_w)
    core_y1 = y1 + int(h * 0.18)
    core_y2 = y1 + int(h * 0.72)

    core = np.zeros_like(upper_region.mask, dtype=np.uint8)
    core[core_y1:core_y2, core_x1:core_x2] = 1
    inner_mask = ((upper_region.mask > 0) & (core > 0)).astype(np.uint8)
    if np.count_nonzero(inner_mask) < 18:
        return None

    if frame is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        upper_pixels = gray[(upper_region.mask > 0)]
        inner_pixels = gray[(inner_mask > 0)]
        if upper_pixels.size > 0 and inner_pixels.size > 0:
            contrast = float(np.mean(inner_pixels) - np.mean(upper_pixels))
            if contrast < 7.0:
                return None

    return MaskRegion(mask=inner_mask, confidence=upper_region.confidence * 0.82)


def detect_outerwear_candidate(
    upper_region: MaskRegion,
    inner_region: MaskRegion | None,
    bbox: tuple[int, int, int, int],
) -> MaskRegion | None:
    """Выделяет внешний слой как область вокруг внутреннего ядра."""
    if inner_region is None:
        return None

    x1, y1, x2, y2 = bbox
    upper = (upper_region.mask > 0).astype(np.uint8)
    inner = (inner_region.mask > 0).astype(np.uint8)
    outer_mask = ((upper == 1) & (inner == 0)).astype(np.uint8)

    if np.count_nonzero(outer_mask) < 30:
        return None

    bbox_area = max(1, (x2 - x1) * (y2 - y1))
    outer_fraction = np.count_nonzero(outer_mask[y1:y2, x1:x2]) / bbox_area
    inner_fraction = np.count_nonzero(inner[y1:y2, x1:x2]) / bbox_area
    if outer_fraction < 0.04 or inner_fraction < 0.01:
        return None

    return MaskRegion(mask=outer_mask, confidence=upper_region.confidence * 0.74)


def build_upper_garments(
    human_id: str,
    bbox: tuple[int, int, int, int],
    hypothesis: UpperGarmentHypothesis,
) -> dict[str, Garment]:
    """Преобразует гипотезу верхней одежды в garment-сущности."""
    garments: dict[str, Garment] = {}
    if hypothesis.upper_inner is not None:
        garment_id = f"{human_id}_garment_upper_inner_0"
        garments[garment_id] = Garment(
            garment_id=garment_id,
            garment_type="upper_inner",
            region=hypothesis.upper_inner,
            visible_fraction=estimate_visible_fraction(hypothesis.upper_inner, bbox),
            state="worn",
            attached_body_parts=["torso", "left_arm", "right_arm"],
            evidence_sources=["parsing"] + (["inner_visible_under_outer_candidate"] if hypothesis.has_layered_upper else []),
            layer_rank=0,
            inferred_only=False,
        )

    if hypothesis.outerwear is not None:
        garment_id = f"{human_id}_garment_outerwear_0"
        garments[garment_id] = Garment(
            garment_id=garment_id,
            garment_type="outerwear",
            region=hypothesis.outerwear,
            visible_fraction=estimate_visible_fraction(hypothesis.outerwear, bbox),
            state="worn",
            attached_body_parts=["torso", "left_arm", "right_arm"],
            evidence_sources=["parsing", "heuristic", "color_contrast"],
            layer_rank=1,
            is_outer_layer_candidate=True,
            inferred_only=True,
        )

    return garments
