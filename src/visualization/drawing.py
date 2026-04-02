from __future__ import annotations

import cv2
import numpy as np

from src.models.schemas import SceneFrame

V1_COLORS: dict[str, tuple[int, int, int]] = {
    "face": (0, 200, 255),
    "hair": (0, 100, 255),
    "upper_clothes": (255, 0, 255),
    "lower_clothes": (255, 255, 0),
    "left_hand": (100, 255, 100),
    "right_hand": (70, 230, 70),
    "left_leg": (255, 100, 100),
    "right_leg": (230, 90, 90),
    "shoes": (255, 255, 255),
}

V2_COLORS: dict[str, tuple[int, int, int]] = {
    "head": (0, 160, 255),
    "face": (0, 210, 255),
    "hair": (40, 110, 255),
    "neck": (0, 240, 220),
    "chest_left": (255, 0, 180),
    "chest_right": (220, 0, 160),
    "abdomen": (255, 160, 0),
    "pelvis": (180, 120, 40),
    "glute_left": (160, 80, 255),
    "glute_right": (140, 60, 240),
    "shoulder_left": (255, 180, 120),
    "shoulder_right": (230, 160, 100),
    "upper_arm_left": (255, 220, 100),
    "upper_arm_right": (230, 200, 90),
    "forearm_left": (255, 240, 120),
    "forearm_right": (230, 220, 100),
    "hand_left": (190, 255, 140),
    "hand_right": (170, 235, 120),
    "thigh_left": (255, 100, 100),
    "thigh_right": (230, 90, 90),
    "knee_left": (255, 125, 120),
    "knee_right": (235, 110, 105),
    "calf_left": (255, 150, 135),
    "calf_right": (235, 140, 125),
    "foot_left": (255, 255, 255),
    "foot_right": (220, 220, 220),
    "back_upper": (120, 170, 255),
    "back_lower": (100, 145, 230),
    "breast_areola": (120, 80, 200),
}

SAM2_COLORS: dict[str, tuple[int, int, int]] = {
    "person_mask": (120, 120, 120),
    "head": (0, 210, 255),
    "torso": (255, 0, 180),
    "left_arm": (255, 220, 100),
    "right_arm": (235, 200, 90),
    "left_leg": (255, 100, 100),
    "right_leg": (230, 90, 90),
}


def draw_detection(scene: SceneFrame) -> np.ndarray:
    """Рисует рамки детекции."""
    canvas = scene.frame.copy()
    for det in scene.detections:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(canvas, f"person {det.confidence:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return canvas


def draw_pose(scene: SceneFrame) -> np.ndarray:
    """Рисует ключевые точки позы."""
    canvas = scene.frame.copy()
    for pose in scene.poses:
        for kp in pose.keypoints:
            if kp.visibility > 0.2:
                cv2.circle(canvas, (int(kp.x), int(kp.y)), 2, (255, 0, 0), -1)
    return canvas


def draw_parsing(scene: SceneFrame) -> np.ndarray:
    """Рисует schema-aware оверлей сегментации."""
    return _draw_parsing_overlay(scene=scene, force_v2_detail=False)


def draw_anatomy_raw_overlay(scene: SceneFrame) -> np.ndarray:
    """Рисует детальный raw anatomy overlay для schema v2 (для v1 — fallback)."""
    return _draw_parsing_overlay(scene=scene, force_v2_detail=True)


def _draw_parsing_overlay(scene: SceneFrame, force_v2_detail: bool) -> np.ndarray:
    """Общий рендер parsing с учетом схемы и режима детализации."""
    canvas = scene.frame.copy()
    for tracked in scene.tracked:
        if tracked.parsed is None:
            continue

        schema = tracked.parsed.schema_version
        is_v2 = schema == "v2"
        is_sam2 = schema == "sam2"
        palette = V2_COLORS if is_v2 else (SAM2_COLORS if is_sam2 else V1_COLORS)
        alpha = 0.45 if (is_v2 and force_v2_detail) else 0.35

        labels = list(tracked.parsed.masks.keys())
        labels.sort(key=lambda name: int(np.count_nonzero(tracked.parsed.masks[name] > 0)), reverse=True)

        for label in labels:
            mask = tracked.parsed.masks[label]
            color = palette.get(label, (180, 180, 180))
            colored = np.zeros_like(canvas)
            colored[:, :] = color
            alpha_mask = (mask > 0).astype(np.uint8)[:, :, None]
            canvas = np.where(alpha_mask == 1, cv2.addWeighted(canvas, 1.0 - alpha, colored, alpha, 0), canvas)

        if is_v2 and force_v2_detail:
            _draw_v2_label_hints(canvas, tracked.parsed.masks, tracked.detection.bbox)
        if is_sam2:
            debug_payload = tracked.parsed.debug if isinstance(tracked.parsed.debug, dict) else {}
            score = float(debug_payload.get("sam2_score", tracked.parsed.confidence))
            x1, y1, _, _ = tracked.detection.bbox
            cv2.putText(
                canvas,
                f"sam2 score={score:.3f}",
                (x1, max(14, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (240, 240, 240),
                1,
            )

    return canvas


def _draw_v2_label_hints(canvas: np.ndarray, masks: dict[str, np.ndarray], bbox: tuple[int, int, int, int]) -> None:
    """Подписывает ключевые anatomy-зоны для визуальной отладки v2."""
    x1, y1, x2, _ = bbox
    y = max(16, y1 - 6)
    labels = ["head", "neck", "chest_left", "chest_right", "abdomen", "pelvis", "upper_arm_left", "upper_arm_right", "thigh_left", "thigh_right"]
    present = [label for label in labels if label in masks and np.count_nonzero(masks[label] > 0) > 0]
    text = "v2 anatomy: " + (", ".join(present[:6]) if present else "no labels")
    cv2.putText(canvas, text, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (245, 245, 245), 1)


def draw_combined(scene: SceneFrame) -> np.ndarray:
    """Рисует объединенную визуализацию."""
    det = draw_detection(scene)
    pose = draw_pose(scene)
    parsing = draw_parsing(scene)
    combined = cv2.addWeighted(det, 0.4, pose, 0.3, 0)
    combined = cv2.addWeighted(combined, 0.7, parsing, 0.3, 0)
    return combined


def draw_sam2_raw_mask(scene: SceneFrame) -> np.ndarray:
    """Отдельный отладочный canvas только с person_mask от SAM2."""
    canvas = scene.frame.copy()
    for tracked in scene.tracked:
        parsed = tracked.parsed
        if parsed is None or parsed.schema_version != "sam2":
            continue
        mask = parsed.masks.get("person_mask")
        if mask is None:
            continue
        color = np.zeros_like(canvas)
        color[:] = (80, 200, 100)
        canvas = np.where((mask > 0)[:, :, None], cv2.addWeighted(canvas, 0.5, color, 0.5, 0), canvas)
    return canvas


def draw_sam2_prompt_debug(scene: SceneFrame) -> np.ndarray:
    """Рисует prompt box/points, отправленные в SAM2."""
    canvas = scene.frame.copy()
    for tracked in scene.tracked:
        parsed = tracked.parsed
        if parsed is None or parsed.schema_version != "sam2":
            continue
        debug_payload = parsed.debug if isinstance(parsed.debug, dict) else {}
        prompt_box = debug_payload.get("prompt_box")
        if isinstance(prompt_box, (list, tuple)) and len(prompt_box) == 4:
            x1, y1, x2, y2 = [int(v) for v in prompt_box]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 2)
        prompt_points = debug_payload.get("prompt_points", [])
        if isinstance(prompt_points, list):
            for point in prompt_points:
                if isinstance(point, list) and len(point) == 2:
                    cv2.circle(canvas, (int(point[0]), int(point[1])), 3, (255, 0, 255), -1)
    return canvas
