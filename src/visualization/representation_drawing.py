from __future__ import annotations

import cv2
import numpy as np

from src.models.schemas import SceneFrame

PART_COLORS: dict[str, tuple[int, int, int]] = {
    "head": (0, 220, 220),
    "head_core": (0, 190, 235),
    "face": (0, 180, 255),
    "hair": (60, 120, 255),
    "neck": (0, 240, 180),
    "torso": (0, 255, 140),
    "left_arm": (90, 210, 255),
    "right_arm": (90, 180, 235),
    "left_leg": (210, 190, 90),
    "right_leg": (180, 160, 70),
    "left_foot": (255, 180, 120),
    "right_foot": (230, 160, 110),
    "chest_left": (255, 50, 200),
    "chest_right": (225, 45, 175),
    "abdomen": (255, 145, 0),
    "pelvis": (190, 120, 40),
    "left_upper_arm": (255, 230, 110),
    "right_upper_arm": (230, 210, 95),
    "left_forearm": (255, 245, 130),
    "right_forearm": (235, 225, 110),
    "left_thigh": (255, 100, 100),
    "right_thigh": (235, 90, 90),
    "left_knee": (255, 130, 120),
    "right_knee": (235, 110, 100),
    "left_calf": (255, 160, 140),
    "right_calf": (240, 145, 125),
}

GARMENT_COLORS: dict[str, tuple[int, int, int]] = {
    "outerwear": (40, 120, 255),
    "upper_inner": (255, 255, 120),
    "pants": (120, 200, 40),
    "shoes": (255, 100, 100),
}

COARSE_PARTS = {"torso", "left_arm", "right_arm", "left_leg", "right_leg", "head", "neck"}


def draw_representation_overlay(scene: SceneFrame) -> np.ndarray:
    """Рисует поверх кадра аккуратное структурированное представление человека."""
    canvas = scene.frame.copy()
    for rep in scene.human_representations:
        x1, y1, x2, y2 = rep.bbox
        _blend_person_mask(canvas, rep.person_mask.mask if rep.person_mask is not None else None, (70, 80, 50), alpha=0.12)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (40, 180, 255), 2)

        layer_text = "/".join(rep.dominant_garments[:2]) if rep.dominant_garments else "no_garments"
        summary = (
            f"{rep.human_id} | {rep.state.pose_state} | layers:{layer_text} "
            f"| p:{rep.reliable_body_parts_count} g:{rep.reliable_garments_count}"
        )
        cv2.putText(canvas, summary, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 180, 255), 1)

        points: dict[str, tuple[int, int]] = {}
        for part in rep.body_parts.values():
            center = _region_center(part.region.mask if part.region is not None else None)
            if center is None:
                continue
            points[part.part_id] = center
            marker_color = (0, 255, 255) if not part.suppressed_from_overlay else (90, 90, 90)
            cv2.circle(canvas, center, 2, marker_color, -1)

        for garment in rep.garments.values():
            center = _region_center(garment.region.mask if garment.region is not None else None)
            if center is None:
                continue
            points[garment.garment_id] = center
            marker_color = (255, 200, 0) if not garment.suppressed_from_overlay else (90, 90, 90)
            cv2.circle(canvas, center, 2, marker_color, -1)

        priority_parts = ("head", "torso", "left_arm", "right_arm", "left_leg", "right_leg", "left_foot", "right_foot")
        label_y = y1 + 14
        for part_id in priority_parts:
            part = rep.body_parts.get(part_id)
            if part is None or part.region is None:
                continue
            if label_y > y2 - 10:
                break
            color = (0, 255, 255) if not part.suppressed_from_overlay else (130, 130, 130)
            cv2.putText(
                canvas,
                f"{part.name}:{part.reliability[0]}{part.reliability_score:.2f}",
                (x1 + 4, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                color,
                1,
            )
            label_y += 11

        garment_label_y = y1 + 14
        for garment in sorted(rep.garments.values(), key=lambda item: item.reliability_score, reverse=True):
            if garment.region is None:
                continue
            if garment_label_y > y2 - 10:
                break
            suffix = "*" if garment.garment_type == "outerwear" else ""
            color = (255, 200, 0) if not garment.suppressed_from_overlay else (130, 130, 130)
            cv2.putText(
                canvas,
                f"{garment.garment_type}{suffix}:{garment.reliability[0]}{garment.reliability_score:.2f}",
                (x2 - 145, garment_label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                color,
                1,
            )
            garment_label_y += 11

        for edge in rep.relations:
            start = points.get(edge.source_id)
            end = points.get(edge.target_id)
            if start is None or end is None or _point_distance(start, end) < 14:
                continue
            cv2.line(canvas, start, end, (120, 120, 255), 1)

    return canvas


def draw_representation_debug(scene: SceneFrame) -> np.ndarray:
    """Строит debug-кадр с текстовой панелью по сущностям representation."""
    frame = scene.frame.copy()
    h, w = frame.shape[:2]
    panel_w = max(520, w // 2)
    panel = np.full((h, panel_w, 3), 30, dtype=np.uint8)

    y = 24
    for rep in scene.human_representations:
        lines = [
            f"{rep.human_id} frame={rep.frame_index} bbox={rep.bbox}",
            f"state={rep.state.pose_state}, L={rep.state.left_arm_state}, R={rep.state.right_arm_state}",
            (
                f"id_conf={rep.identity_confidence:.2f}, parse_conf={rep.parsing_confidence:.2f}, "
                f"layered={rep.has_layered_upper_clothing}"
            ),
            f"dominant={','.join(rep.dominant_garments) if rep.dominant_garments else 'none'}",
            "parts:",
        ]
        for part in rep.body_parts.values():
            if part.region is None:
                continue
            lines.append(
                "  - "
                f"{part.name}: vis={part.visible_fraction:.3f} rel={part.reliability}/{part.reliability_score:.2f} "
                f"ev={'+'.join(part.evidence_sources)} sup_o={part.suppressed_from_overlay} "
                f"sup_r={part.suppressed_from_relations} inf={part.inferred_only}"
            )
        lines.append("garments:")
        for garment in rep.garments.values():
            if garment.region is None:
                continue
            lines.append(
                "  - "
                f"{garment.garment_id}: {garment.garment_type} vis={garment.visible_fraction:.3f} "
                f"rel={garment.reliability}/{garment.reliability_score:.2f} "
                f"layer={garment.layer_rank} outer={garment.is_outer_layer_candidate} "
                f"ev={'+'.join(garment.evidence_sources)} sup_o={garment.suppressed_from_overlay} "
                f"sup_r={garment.suppressed_from_relations} inf={garment.inferred_only}"
            )
        lines.append("relations:")
        for rel in rep.relations:
            lines.append(f"  - {rel.source_id} {rel.relation_type} {rel.target_id} ({rel.confidence:.2f})")

        for line in lines:
            if y > h - 14:
                break
            cv2.putText(panel, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
            y += 16
        y += 8

    return np.hstack([frame, panel])


def draw_representation_masks(scene: SceneFrame) -> np.ndarray:
    """Совместимый рендер масок representation (эквивалент normalized)."""
    return draw_representation_masks_normalized(scene)


def draw_representation_masks_raw(scene: SceneFrame) -> np.ndarray:
    """Рисует raw body-part маски (fine-grained, включая подавленные бледно)."""
    canvas = scene.frame.copy()
    for rep in scene.human_representations:
        if rep.person_mask is not None:
            _blend_person_mask(canvas, rep.person_mask.mask, (80, 80, 80), alpha=0.1)
        for part in rep.body_parts.values():
            if part.region is None:
                continue
            color = PART_COLORS.get(part.name, (120, 120, 120))
            alpha = 0.18 if part.suppressed_from_overlay else 0.28
            _blend_person_mask(canvas, part.region.mask, color, alpha=alpha)
            _draw_contours(canvas, part.region.mask, color)
    return _draw_masks_legend(canvas, mode="raw")


def draw_representation_masks_normalized(scene: SceneFrame) -> np.ndarray:
    """Рисует нормализованные coarse body parts для reasoning-слоя."""
    canvas = scene.frame.copy()
    for rep in scene.human_representations:
        if rep.person_mask is not None:
            _blend_person_mask(canvas, rep.person_mask.mask, (80, 80, 80), alpha=0.08)
        for part in rep.body_parts.values():
            if part.region is None or part.name not in COARSE_PARTS:
                continue
            color = PART_COLORS.get(part.name, (120, 120, 120))
            alpha = 0.16 if part.suppressed_from_overlay else 0.32
            _blend_person_mask(canvas, part.region.mask, color, alpha=alpha)
            _draw_contours(canvas, part.region.mask, color)
    return _draw_masks_legend(canvas, mode="normalized")


def draw_representation_masks_garments(scene: SceneFrame) -> np.ndarray:
    """Рисует garment-семантику и слои верхней одежды."""
    canvas = scene.frame.copy()
    for rep in scene.human_representations:
        for garment in rep.garments.values():
            if garment.region is None:
                continue
            color = GARMENT_COLORS.get(garment.garment_type, (200, 200, 200))
            alpha = 0.18 if garment.suppressed_from_overlay else 0.36
            _blend_person_mask(canvas, garment.region.mask, color, alpha=alpha)
            _draw_contours(canvas, garment.region.mask, color)

            center = _region_center(garment.region.mask)
            if center is not None:
                label = f"{garment.garment_type} {garment.reliability_score:.2f}"
                cv2.putText(canvas, label, center, cv2.FONT_HERSHEY_SIMPLEX, 0.36, (250, 250, 250), 1)
    return _draw_masks_legend(canvas, mode="garments")


def draw_summary_panel(images: dict[str, np.ndarray]) -> np.ndarray:
    """Собирает информативный summary_panel для anatomy и legacy режимов."""
    base = images.get("combined")
    if base is None:
        raise ValueError("Для summary_panel требуется изображение combined")

    h, w = base.shape[:2]
    tiles = [
        _draw_tile_caption(_fit_with_padding(images.get("detection", base), (w, h)), "detection"),
        _draw_tile_caption(_fit_with_padding(images.get("parsing", base), (w, h)), "parsing"),
        _draw_tile_caption(_fit_with_padding(images.get("anatomy_raw_overlay", images.get("parsing", base)), (w, h)), "anatomy_raw_overlay"),
        _draw_tile_caption(_fit_with_padding(images.get("representation_masks_raw", base), (w, h)), "representation_masks_raw"),
        _draw_tile_caption(_fit_with_padding(images.get("representation_masks_normalized", base), (w, h)), "representation_masks_normalized"),
        _draw_tile_caption(_fit_with_padding(images.get("representation_masks_garments", base), (w, h)), "representation_masks_garments"),
        _draw_tile_caption(_fit_with_padding(images.get("representation_overlay", base), (w, h)), "representation_overlay"),
        _draw_tile_caption(_fit_with_padding(base, (w, h)), "combined"),
        _draw_tile_caption(_debug_preview(images.get("representation_debug", base), (w, h), frame_width=w), "representation_debug_preview"),
    ]
    return np.vstack([np.hstack(tiles[0:3]), np.hstack(tiles[3:6]), np.hstack(tiles[6:9])])


def _draw_tile_caption(image: np.ndarray, title: str) -> np.ndarray:
    """Добавляет подпись плитки на копии изображения."""
    tile = image.copy()
    cv2.putText(tile, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
    return tile


def _fit_with_padding(image: np.ndarray, target_wh: tuple[int, int]) -> np.ndarray:
    """Масштабирует изображение с сохранением пропорций и полями."""
    target_w, target_h = target_wh
    src_h, src_w = image.shape[:2]
    if src_h <= 0 or src_w <= 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(src_w * scale))
    new_h = max(1, int(src_h * scale))
    resized = cv2.resize(image, (new_w, new_h))

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2
    canvas[y : y + new_h, x : x + new_w] = resized
    return canvas


def _debug_preview(debug_image: np.ndarray, target_wh: tuple[int, int], frame_width: int) -> np.ndarray:
    """Строит читаемый preview для широкой debug-картинки."""
    target_w, target_h = target_wh
    h, w = debug_image.shape[:2]
    if w <= frame_width:
        return _fit_with_padding(debug_image, target_wh)

    left = debug_image[:, :frame_width]
    right = debug_image[:, frame_width:]
    left_width = max(1, target_w // 3)
    right_width = target_w - left_width

    right_crop_h = max(1, int(h * 0.7))
    right_crop = right[:right_crop_h, :]

    left_small = cv2.resize(left, (left_width, target_h))
    right_small = _fit_with_padding(right_crop, (right_width, target_h))
    return np.hstack([left_small, right_small])


def _draw_masks_legend(canvas: np.ndarray, mode: str) -> np.ndarray:
    """Рисует легенду семантических масок под конкретный режим."""
    legend = canvas.copy()
    items = {
        "raw": [
            ("person_mask", (80, 80, 80)),
            ("fine anatomy", PART_COLORS["chest_left"]),
            ("coarse anatomy", PART_COLORS["torso"]),
            ("suppressed=fade", (100, 100, 100)),
        ],
        "normalized": [
            ("torso", PART_COLORS["torso"]),
            ("head", PART_COLORS["head"]),
            ("arms", PART_COLORS["left_arm"]),
            ("legs", PART_COLORS["left_leg"]),
        ],
        "garments": [
            ("outerwear", GARMENT_COLORS["outerwear"]),
            ("upper_inner", GARMENT_COLORS["upper_inner"]),
            ("pants", GARMENT_COLORS["pants"]),
            ("shoes", GARMENT_COLORS["shoes"]),
        ],
    }.get(mode, [])

    x, y = 12, 16
    for label, color in items:
        cv2.rectangle(legend, (x, y - 10), (x + 12, y + 2), color, -1)
        cv2.putText(legend, label, (x + 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (245, 245, 245), 1)
        y += 16
    return legend


def _draw_contours(canvas: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> None:
    """Рисует контуры бинарной маски."""
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(canvas, contours, -1, color, 1)


def _region_center(mask: np.ndarray | None) -> tuple[int, int] | None:
    """Возвращает центр бинарного региона или None."""
    if mask is None:
        return None
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(np.mean(xs)), int(np.mean(ys))


def _blend_person_mask(canvas: np.ndarray, mask: np.ndarray | None, color: tuple[int, int, int], alpha: float) -> None:
    """Полупрозрачно подсвечивает маску без захламления кадра."""
    if mask is None or np.count_nonzero(mask) == 0:
        return
    overlay = np.zeros_like(canvas)
    overlay[:, :] = color
    mask3 = (mask > 0).astype(np.uint8)[:, :, None]
    blended = cv2.addWeighted(canvas, 1.0 - alpha, overlay, alpha, 0)
    np.copyto(canvas, np.where(mask3 == 1, blended, canvas))


def _point_distance(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Считает расстояние между двумя точками на плоскости."""
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))
