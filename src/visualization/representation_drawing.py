from __future__ import annotations

import cv2
import numpy as np

from src.models.schemas import SceneFrame


def draw_representation_overlay(scene: SceneFrame) -> np.ndarray:
    """Рисует поверх кадра аккуратное структурированное представление человека."""
    canvas = scene.frame.copy()
    for rep in scene.human_representations:
        x1, y1, x2, y2 = rep.bbox
        _blend_person_mask(canvas, rep.person_mask.mask if rep.person_mask is not None else None, (70, 80, 50), alpha=0.12)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (40, 180, 255), 2)

        summary = f"{rep.human_id} | {rep.state.pose_state} L:{rep.state.left_arm_state[0]} R:{rep.state.right_arm_state[0]}"
        cv2.putText(canvas, summary, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 180, 255), 1)

        points: dict[str, tuple[int, int]] = {}
        for part in rep.body_parts.values():
            center = _region_center(part.region.mask if part.region is not None else None)
            if center is None:
                continue
            points[part.part_id] = center
            cv2.circle(canvas, center, 2, (0, 255, 255), -1)

        for garment in rep.garments.values():
            center = _region_center(garment.region.mask if garment.region is not None else None)
            if center is None:
                continue
            points[garment.garment_id] = center
            cv2.circle(canvas, center, 2, (255, 200, 0), -1)

        priority_parts = ("head", "torso", "left_arm", "right_arm", "left_leg", "right_leg", "left_foot", "right_foot")
        label_y = y1 + 14
        for part_id in priority_parts:
            part = rep.body_parts.get(part_id)
            if part is None or part.region is None or part.visible_fraction <= 0.01:
                continue
            if label_y > y2 - 10:
                break
            cv2.putText(canvas, part.name, (x1 + 4, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 255, 255), 1)
            label_y += 11

        garment_label_y = y1 + 14
        seen_garments: set[str] = set()
        for garment in rep.garments.values():
            if garment.region is None or garment.visible_fraction <= 0.01 or garment.garment_type in seen_garments:
                continue
            if garment_label_y > y2 - 10:
                break
            seen_garments.add(garment.garment_type)
            cv2.putText(canvas, garment.garment_type, (x2 - 95, garment_label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 200, 0), 1)
            garment_label_y += 11

        for edge in rep.relations:
            start = points.get(edge.source_id)
            end = points.get(edge.target_id)
            if start is None or end is None:
                continue
            if _point_distance(start, end) < 14:
                continue
            cv2.line(canvas, start, end, (120, 120, 255), 1)

    return canvas


def draw_representation_debug(scene: SceneFrame) -> np.ndarray:
    """Строит debug-кадр с текстовой панелью по сущностям representation."""
    frame = scene.frame.copy()
    h, w = frame.shape[:2]
    panel_w = max(380, w // 2)
    panel = np.full((h, panel_w, 3), 30, dtype=np.uint8)

    y = 24
    for rep in scene.human_representations:
        lines = [
            f"{rep.human_id} frame={rep.frame_index} bbox={rep.bbox}",
            f"state={rep.state.pose_state}, L={rep.state.left_arm_state}, R={rep.state.right_arm_state}",
            f"id_conf={rep.identity_confidence:.2f}, parse_conf={rep.parsing_confidence:.2f}",
            "parts:",
        ]
        for part in rep.body_parts.values():
            if part.region is None:
                continue
            lines.append(f"  - {part.name}: vis={part.visible_fraction:.2f}, occ={part.occluded}")
        lines.append("garments:")
        for garment in rep.garments.values():
            if garment.region is None:
                continue
            lines.append(f"  - {garment.garment_id}: {garment.garment_type}, vis={garment.visible_fraction:.2f}")
        lines.append("relations:")
        for rel in rep.relations:
            lines.append(f"  - {rel.source_id} {rel.relation_type} {rel.target_id} ({rel.confidence:.2f})")

        for line in lines:
            if y > h - 14:
                break
            cv2.putText(panel, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
            y += 16
        y += 8

    return np.hstack([frame, panel])


def draw_summary_panel(images: dict[str, np.ndarray]) -> np.ndarray:
    """Собирает контакт-лист ключевых визуализаций без агрессивного сжатия debug-панели."""
    base = images.get("combined")
    if base is None:
        raise ValueError("Для summary_panel требуется изображение combined")

    h, w = base.shape[:2]
    detection = _fit_with_padding(images.get("detection", base), (w, h))
    parsing = _fit_with_padding(images.get("parsing", base), (w, h))
    combined = _fit_with_padding(base, (w, h))
    overlay = _fit_with_padding(images.get("representation_overlay", base), (w, h))

    debug_full = images.get("representation_debug", base)
    debug_preview = _debug_preview(debug_full, (w, h), frame_width=w)

    tiles = [
        _draw_tile_caption(detection, "detection"),
        _draw_tile_caption(parsing, "parsing"),
        _draw_tile_caption(combined, "combined"),
        _draw_tile_caption(overlay, "representation_overlay"),
        _draw_tile_caption(debug_preview, "representation_debug_preview"),
        np.zeros((h, w, 3), dtype=np.uint8),
    ]
    return np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:6])])


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

    # Для читаемости текста берем верхнюю часть правой панели и отдаем ей большую ширину.
    right_crop_h = max(1, int(h * 0.7))
    right_crop = right[:right_crop_h, :]

    left_small = cv2.resize(left, (left_width, target_h))
    right_small = _fit_with_padding(right_crop, (right_width, target_h))
    return np.hstack([left_small, right_small])


def _region_center(mask: np.ndarray | None) -> tuple[int, int] | None:
    """Возвращает центр бинарного региона или None."""
    if mask is None:
        return None
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(np.mean(xs)), int(np.mean(ys))


def _blend_person_mask(canvas: np.ndarray, mask: np.ndarray | None, color: tuple[int, int, int], alpha: float) -> None:
    """Полупрозрачно подсвечивает person_mask без захламления кадра."""
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
