from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.models.schemas import ParsedHuman
from src.representation.garment_reasoning import (
    infer_upper_garment_candidates_v1,
    infer_upper_garment_candidates_v2,
)
from src.representation.schemas import BodyPart, Garment, Keypoint2D, MaskRegion
from src.representation.state_rules import estimate_visible_fraction, sanitize_bbox


@dataclass(slots=True)
class _AdapterContext:
    """Контекст построения сущностей representation для одной схемы парсинга."""

    human_id: str
    bbox: tuple[int, int, int, int]
    shape: tuple[int, int]
    parsing: ParsedHuman
    keypoints: dict[str, Keypoint2D]
    frame: np.ndarray | None

    @property
    def model_evidence(self) -> list[str]:
        """Возвращает маркеры источника модели для калибровки confidence."""
        if "stub" in self.parsing.model_version:
            return [self.parsing.model_version, "sam2-anatomy-stub"]
        return [self.parsing.model_version]


class V1ParsingAdapter:
    """Адаптер lifting-логики для legacy schema v1."""

    def build_body_parts(self, ctx: _AdapterContext, *, min_kp_conf: float, arm_thickness_ratio: float) -> dict[str, BodyPart]:
        masks = ctx.parsing.masks
        face_region = _safe_region(masks.get("face"), ctx.parsing)
        hair_region = _safe_region(masks.get("hair"), ctx.parsing)
        head_region = _union_regions([face_region, hair_region], ["parsing", "union"]) 

        parts: dict[str, BodyPart] = {}
        region_specs: dict[str, tuple[MaskRegion | None, list[str], bool]] = {
            "head": (_clip_to_bbox(head_region, ctx.bbox, ctx.shape), ["parsing", "union"], False),
            "face": (_clip_to_bbox(face_region, ctx.bbox, ctx.shape), ["parsing"], False),
            "hair": (_clip_to_bbox(hair_region, ctx.bbox, ctx.shape), ["parsing"], False),
            "torso": (_clip_to_bbox(_safe_region(masks.get("upper_clothes"), ctx.parsing), ctx.bbox, ctx.shape), ["parsing"], False),
            "left_hand": (_clip_to_bbox(_safe_region(masks.get("left_hand"), ctx.parsing), ctx.bbox, ctx.shape), ["parsing"], False),
            "right_hand": (_clip_to_bbox(_safe_region(masks.get("right_hand"), ctx.parsing), ctx.bbox, ctx.shape), ["parsing"], False),
            "left_leg": (_clip_to_bbox(_safe_region(masks.get("left_leg"), ctx.parsing), ctx.bbox, ctx.shape), ["parsing"], False),
            "right_leg": (_clip_to_bbox(_safe_region(masks.get("right_leg"), ctx.parsing), ctx.bbox, ctx.shape), ["parsing"], False),
        }

        arm_regions = _build_arm_regions(
            keypoints=ctx.keypoints,
            bbox=ctx.bbox,
            shape=next(iter(masks.values())).shape[:2] if masks else ctx.shape,
            min_keypoint_confidence=min_kp_conf,
            arm_thickness_ratio=arm_thickness_ratio,
        )
        for arm_name, arm_region in arm_regions.items():
            if arm_region is not None:
                region_specs[arm_name] = (_clip_to_bbox(arm_region, ctx.bbox, ctx.shape), ["pose", "heuristic"], True)

        left_foot, right_foot = _split_shoes_region(masks.get("shoes"), ctx.parsing, ctx.bbox)
        if left_foot is not None:
            region_specs["left_foot"] = (_clip_to_bbox(left_foot, ctx.bbox, ctx.shape), ["parsing", "split_from_shoes"], True)
        if right_foot is not None:
            region_specs["right_foot"] = (_clip_to_bbox(right_foot, ctx.bbox, ctx.shape), ["parsing", "split_from_shoes"], True)

        if "left_shoulder" in ctx.keypoints and "right_shoulder" in ctx.keypoints:
            neck_region = _build_neck_region(ctx.bbox, ctx.parsing)
            if neck_region is not None:
                region_specs["neck"] = (_clip_to_bbox(neck_region, ctx.bbox, ctx.shape), ["heuristic", "pose"], True)

        for part_name, (region, evidence_sources, inferred_only) in region_specs.items():
            parts[part_name] = _make_part(part_name, region, evidence_sources, inferred_only, ctx.bbox)

        return parts

    def build_garments(self, ctx: _AdapterContext) -> dict[str, Garment]:
        garments: dict[str, Garment] = {}
        hypothesis = infer_upper_garment_candidates_v1(frame=ctx.frame, bbox=ctx.bbox, parsing=ctx.parsing)
        garments.update(
            _build_upper_garments(ctx.human_id, ctx.bbox, hypothesis, schema_version="v1", model_evidence=ctx.model_evidence)
        )

        lower = _clip_to_bbox(_safe_region(ctx.parsing.masks.get("lower_clothes"), ctx.parsing), ctx.bbox, ctx.shape)
        if lower is not None:
            garments[f"{ctx.human_id}_garment_lower_0"] = Garment(
                garment_id=f"{ctx.human_id}_garment_lower_0",
                garment_type="pants",
                region=lower,
                visible_fraction=estimate_visible_fraction(lower, ctx.bbox),
                state="worn",
                attached_body_parts=["left_leg", "right_leg"],
                evidence_sources=["parsing", "schema_v1", *ctx.model_evidence, "clipped_to_bbox"],
            )

        shoes = _clip_to_bbox(_safe_region(ctx.parsing.masks.get("shoes"), ctx.parsing), ctx.bbox, ctx.shape)
        if shoes is not None:
            garments[f"{ctx.human_id}_garment_shoes_0"] = Garment(
                garment_id=f"{ctx.human_id}_garment_shoes_0",
                garment_type="shoes",
                region=shoes,
                visible_fraction=estimate_visible_fraction(shoes, ctx.bbox),
                state="worn",
                attached_body_parts=["left_foot", "right_foot"],
                evidence_sources=["parsing", "schema_v1", *ctx.model_evidence, "clipped_to_bbox"],
            )

        return garments


class V2ParsingAdapter:
    """Адаптер lifting-логики для anatomy schema v2."""

    V2_TO_PART: dict[str, str] = {
        "head": "head_core",
        "face": "face",
        "hair": "hair",
        "neck": "neck",
        "chest_left": "chest_left",
        "chest_right": "chest_right",
        "abdomen": "abdomen",
        "pelvis": "pelvis",
        "glute_left": "glute_left",
        "glute_right": "glute_right",
        "back_upper": "back_upper",
        "back_lower": "back_lower",
        "breast_areola": "breast_areola",
        "shoulder_left": "left_shoulder",
        "shoulder_right": "right_shoulder",
        "upper_arm_left": "left_upper_arm",
        "upper_arm_right": "right_upper_arm",
        "forearm_left": "left_forearm",
        "forearm_right": "right_forearm",
        "hand_left": "left_hand",
        "hand_right": "right_hand",
        "thigh_left": "left_thigh",
        "thigh_right": "right_thigh",
        "knee_left": "left_knee",
        "knee_right": "right_knee",
        "calf_left": "left_calf",
        "calf_right": "right_calf",
        "foot_left": "left_foot",
        "foot_right": "right_foot",
    }

    def build_body_parts(self, ctx: _AdapterContext, *, min_kp_conf: float, arm_thickness_ratio: float) -> dict[str, BodyPart]:
        del min_kp_conf, arm_thickness_ratio
        parts: dict[str, BodyPart] = {}

        for label, part_name in self.V2_TO_PART.items():
            region = _clip_to_bbox(_safe_region(ctx.parsing.masks.get(label), ctx.parsing, per_label=True, label=label), ctx.bbox, ctx.shape)
            inferred = label not in ctx.parsing.masks or np.count_nonzero(ctx.parsing.masks.get(label, 0)) == 0
            parts[part_name] = _make_part(
                part_name,
                region,
                ["parsing", "schema_v2", f"label:{label}", *ctx.model_evidence],
                inferred,
                ctx.bbox,
            )

        # Нормализованные coarse-сущности из fine anatomy.
        coarse_specs: dict[str, tuple[list[str], list[str]]] = {
            "torso": (["chest_left", "chest_right", "abdomen", "pelvis"], ["parsing", "schema_v2", "union", "coarse"]),
            "left_arm": (["left_shoulder", "left_upper_arm", "left_forearm", "left_hand"], ["parsing", "schema_v2", "union", "coarse"]),
            "right_arm": (["right_shoulder", "right_upper_arm", "right_forearm", "right_hand"], ["parsing", "schema_v2", "union", "coarse"]),
            "left_leg": (["left_thigh", "left_knee", "left_calf", "left_foot"], ["parsing", "schema_v2", "union", "coarse"]),
            "right_leg": (["right_thigh", "right_knee", "right_calf", "right_foot"], ["parsing", "schema_v2", "union", "coarse"]),
            "head": (["head_core", "face", "hair"], ["parsing", "schema_v2", "union", "coarse"]),
        }

        for part_name, (source_parts, evidence) in coarse_specs.items():
            source_regions = [parts[p].region for p in source_parts if p in parts]
            union_region = _union_regions(source_regions, evidence)
            parts[part_name] = _make_part(
                part_name,
                _clip_to_bbox(union_region, ctx.bbox, ctx.shape),
                [*evidence, *ctx.model_evidence],
                True,
                ctx.bbox,
            )

        return parts

    def build_garments(self, ctx: _AdapterContext) -> dict[str, Garment]:
        garments: dict[str, Garment] = {}
        hypothesis = infer_upper_garment_candidates_v2(frame=ctx.frame, bbox=ctx.bbox, parsing=ctx.parsing)
        garments.update(
            _build_upper_garments(ctx.human_id, ctx.bbox, hypothesis, schema_version="v2", model_evidence=ctx.model_evidence)
        )

        leg_union = _union_regions(
            [
                _safe_region(ctx.parsing.masks.get("pelvis"), ctx.parsing, per_label=True, label="pelvis"),
                _safe_region(ctx.parsing.masks.get("thigh_left"), ctx.parsing, per_label=True, label="thigh_left"),
                _safe_region(ctx.parsing.masks.get("thigh_right"), ctx.parsing, per_label=True, label="thigh_right"),
                _safe_region(ctx.parsing.masks.get("knee_left"), ctx.parsing, per_label=True, label="knee_left"),
                _safe_region(ctx.parsing.masks.get("knee_right"), ctx.parsing, per_label=True, label="knee_right"),
                _safe_region(ctx.parsing.masks.get("calf_left"), ctx.parsing, per_label=True, label="calf_left"),
                _safe_region(ctx.parsing.masks.get("calf_right"), ctx.parsing, per_label=True, label="calf_right"),
            ],
            ["parsing", "schema_v2", "union", "anatomy_anchor"],
        )
        leg_union = _clip_to_bbox(leg_union, ctx.bbox, ctx.shape)
        if leg_union is not None:
            garments[f"{ctx.human_id}_garment_lower_0"] = Garment(
                garment_id=f"{ctx.human_id}_garment_lower_0",
                garment_type="pants",
                region=leg_union,
                visible_fraction=estimate_visible_fraction(leg_union, ctx.bbox),
                state="worn",
                attached_body_parts=["pelvis", "left_thigh", "right_thigh", "left_leg", "right_leg"],
                evidence_sources=["parsing", "schema_v2", "union", "anatomy_anchor", *ctx.model_evidence, "clipped_to_bbox"],
                inferred_only=True,
            )

        shoes = _union_regions(
            [
                _safe_region(ctx.parsing.masks.get("foot_left"), ctx.parsing, per_label=True, label="foot_left"),
                _safe_region(ctx.parsing.masks.get("foot_right"), ctx.parsing, per_label=True, label="foot_right"),
            ],
            ["parsing", "schema_v2", "union", "anatomy_anchor"],
        )
        shoes = _clip_to_bbox(shoes, ctx.bbox, ctx.shape)
        if shoes is not None:
            garments[f"{ctx.human_id}_garment_shoes_0"] = Garment(
                garment_id=f"{ctx.human_id}_garment_shoes_0",
                garment_type="shoes",
                region=shoes,
                visible_fraction=estimate_visible_fraction(shoes, ctx.bbox),
                state="worn",
                attached_body_parts=["left_foot", "right_foot"],
                evidence_sources=["parsing", "schema_v2", "union", "anatomy_anchor", *ctx.model_evidence, "clipped_to_bbox"],
                inferred_only=True,
            )

        return garments


class SAM2ParsingAdapter:
    """Адаптер для schema sam2: person mask + coarse части тела."""

    PARTS: dict[str, str] = {
        "head": "head",
        "torso": "torso",
        "left_arm": "left_arm",
        "right_arm": "right_arm",
        "left_leg": "left_leg",
        "right_leg": "right_leg",
    }

    def build_body_parts(self, ctx: _AdapterContext, *, min_kp_conf: float, arm_thickness_ratio: float) -> dict[str, BodyPart]:
        del min_kp_conf, arm_thickness_ratio
        parts: dict[str, BodyPart] = {}
        for label, part_name in self.PARTS.items():
            region = _clip_to_bbox(_safe_region(ctx.parsing.masks.get(label), ctx.parsing, per_label=True, label=label), ctx.bbox, ctx.shape)
            parts[part_name] = _make_part(
                part_name,
                region,
                ["parsing", "schema_sam2", "heuristic", f"label:{label}", *ctx.model_evidence],
                True,
                ctx.bbox,
            )
        return parts

    def build_garments(self, ctx: _AdapterContext) -> dict[str, Garment]:
        del ctx
        # На SAM2 path не создаем garment labels автоматически.
        # Причина: текущая ветка SAM2 дает надежную person segmentation,
        # но не дает надежных garment semantics.
        return {}


def create_adapter(schema_version: str) -> V1ParsingAdapter | V2ParsingAdapter | SAM2ParsingAdapter:
    """Возвращает адаптер под schema_version или бросает ошибку для неизвестной схемы."""
    if schema_version == "v1":
        return V1ParsingAdapter()
    if schema_version == "v2":
        return V2ParsingAdapter()
    if schema_version == "sam2":
        return SAM2ParsingAdapter()
    raise ValueError(f"Неизвестная schema_version: {schema_version}")


def build_context(
    *,
    human_id: str,
    bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
    parsing: ParsedHuman,
    keypoints: dict[str, Keypoint2D],
    frame: np.ndarray | None,
) -> _AdapterContext:
    """Строит внутренний контекст адаптера."""
    return _AdapterContext(human_id=human_id, bbox=bbox, shape=shape, parsing=parsing, keypoints=keypoints, frame=frame)


def _make_part(
    part_name: str,
    region: MaskRegion | None,
    evidence_sources: list[str],
    inferred_only: bool,
    bbox: tuple[int, int, int, int],
) -> BodyPart:
    clipped_evidence = [*evidence_sources, "clipped_to_bbox"] if region is not None else list(evidence_sources)
    visible_fraction = estimate_visible_fraction(region=region, bbox=bbox)
    return BodyPart(
        part_id=part_name,
        name=part_name,
        region=region,
        visible_fraction=visible_fraction,
        occluded=(region is None) or visible_fraction < 0.1,
        evidence_sources=clipped_evidence,
        inferred_only=inferred_only,
    )


def _safe_region(mask: np.ndarray | None, parsing: ParsedHuman, *, per_label: bool = False, label: str | None = None) -> MaskRegion | None:
    """Преобразует бинарную маску в region с учетом уверенности конкретного лейбла."""
    if mask is None:
        return None
    confidence = float(parsing.confidence)
    if per_label and label is not None:
        confidence = float(parsing.label_confidence.get(label, parsing.confidence))
    return MaskRegion(mask=(mask > 0).astype(np.uint8), confidence=confidence)


def _union_regions(regions: list[MaskRegion | None], evidence: list[str]) -> MaskRegion | None:
    """Объединяет регионы и усредняет уверенность по валидным источникам."""
    del evidence
    valid = [r for r in regions if r is not None and np.count_nonzero(r.mask) > 0]
    if not valid:
        return None
    out = valid[0].mask.copy().astype(np.uint8)
    confidences: list[float] = [valid[0].confidence]
    for reg in valid[1:]:
        out = np.maximum(out, (reg.mask > 0).astype(np.uint8))
        confidences.append(reg.confidence)
    return MaskRegion(mask=out, confidence=float(np.mean(confidences)))


def _clip_to_bbox(region: MaskRegion | None, bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> MaskRegion | None:
    """Обрезает region рамкой человека."""
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


def _build_arm_regions(
    keypoints: dict[str, Keypoint2D],
    bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
    min_keypoint_confidence: float,
    arm_thickness_ratio: float,
) -> dict[str, MaskRegion | None]:
    """Строит эвристические arm-маски по позе."""
    return {
        "left_arm": _draw_arm_mask("left", keypoints, bbox, shape, min_keypoint_confidence, arm_thickness_ratio),
        "right_arm": _draw_arm_mask("right", keypoints, bbox, shape, min_keypoint_confidence, arm_thickness_ratio),
    }


def _draw_arm_mask(
    side: str,
    keypoints: dict[str, Keypoint2D],
    bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
    min_keypoint_confidence: float,
    arm_thickness_ratio: float,
) -> MaskRegion | None:
    """Рисует маску руки по ключевым точкам."""
    shoulder = keypoints.get(f"{side}_shoulder")
    elbow = keypoints.get(f"{side}_elbow")
    wrist = keypoints.get(f"{side}_wrist")
    if shoulder is None or wrist is None:
        return None
    if shoulder.confidence < min_keypoint_confidence or wrist.confidence < min_keypoint_confidence:
        return None

    height = max(1, bbox[3] - bbox[1])
    thickness = max(2, int(height * arm_thickness_ratio))
    arm_mask = np.zeros(shape, dtype=np.uint8)
    pts = [(int(shoulder.x), int(shoulder.y))]
    if elbow is not None and elbow.confidence >= min_keypoint_confidence:
        pts.append((int(elbow.x), int(elbow.y)))
    pts.append((int(wrist.x), int(wrist.y)))
    for i in range(len(pts) - 1):
        cv2.line(arm_mask, pts[i], pts[i + 1], 1, thickness=thickness)
    if np.count_nonzero(arm_mask) == 0:
        return None
    return MaskRegion(mask=arm_mask, confidence=0.6)


def _split_shoes_region(
    shoes_mask: np.ndarray | None,
    parsing: ParsedHuman,
    bbox: tuple[int, int, int, int],
) -> tuple[MaskRegion | None, MaskRegion | None]:
    """Разделяет общую маску обуви на левую и правую."""
    if shoes_mask is None:
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


def _build_neck_region(bbox: tuple[int, int, int, int], parsing: ParsedHuman) -> MaskRegion | None:
    """Строит эвристический neck-region внутри bbox."""
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


def _build_upper_garments(
    human_id: str,
    bbox: tuple[int, int, int, int],
    hypothesis,
    *,
    schema_version: str,
    model_evidence: list[str],
) -> dict[str, Garment]:
    """Преобразует гипотезу верхней одежды в Garment-сущности."""
    garments: dict[str, Garment] = {}
    attached_body_parts = ["torso", "left_arm", "right_arm"]
    if schema_version == "v2":
        attached_body_parts = ["torso", "left_upper_arm", "right_upper_arm", "left_arm", "right_arm"]

    if hypothesis.upper_inner is not None:
        garment_id = f"{human_id}_garment_upper_inner_0"
        garments[garment_id] = Garment(
            garment_id=garment_id,
            garment_type="upper_inner",
            region=hypothesis.upper_inner,
            visible_fraction=estimate_visible_fraction(hypothesis.upper_inner, bbox),
            state="worn",
            attached_body_parts=attached_body_parts,
            evidence_sources=["parsing", f"schema_{schema_version}", *model_evidence]
            + (["inner_visible_under_outer_candidate"] if hypothesis.has_layered_upper else [])
            + list(hypothesis.evidence),
            layer_rank=0,
            inferred_only=hypothesis.has_layered_upper,
        )

    if hypothesis.outerwear is not None:
        garment_id = f"{human_id}_garment_outerwear_0"
        garments[garment_id] = Garment(
            garment_id=garment_id,
            garment_type="outerwear",
            region=hypothesis.outerwear,
            visible_fraction=estimate_visible_fraction(hypothesis.outerwear, bbox),
            state="worn",
            attached_body_parts=attached_body_parts,
            evidence_sources=["parsing", "heuristic", f"schema_{schema_version}", *model_evidence] + list(hypothesis.evidence),
            layer_rank=1,
            is_outer_layer_candidate=True,
            inferred_only=True,
        )

    return garments
