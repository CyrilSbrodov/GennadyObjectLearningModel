from __future__ import annotations

import cv2
import numpy as np

from src.models.schemas import SceneFrame


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
    """Рисует оверлей сегментации."""
    canvas = scene.frame.copy()
    colors = {
        "face": (0, 200, 255),
        "hair": (0, 100, 255),
        "head": (0, 160, 255),
        "neck": (0, 220, 220),
        "chest_left": (255, 0, 180),
        "chest_right": (220, 0, 160),
        "abdomen": (255, 160, 0),
        "pelvis": (180, 120, 40),
        "glute_left": (160, 80, 255),
        "glute_right": (140, 60, 240),
        "thigh_left": (255, 100, 100),
        "thigh_right": (240, 90, 90),
        "calf_left": (255, 140, 120),
        "calf_right": (240, 130, 110),
        "foot_left": (255, 255, 255),
        "foot_right": (220, 220, 220),
        "upper_clothes": (255, 0, 255),
        "lower_clothes": (255, 255, 0),
        "left_hand": (100, 255, 100),
        "right_hand": (100, 255, 100),
        "left_leg": (255, 100, 100),
        "right_leg": (255, 100, 100),
        "shoes": (255, 255, 255),
    }
    for tracked in scene.tracked:
        if tracked.parsed is None:
            continue
        for label, mask in tracked.parsed.masks.items():
            color = colors.get(label, (180, 180, 180))
            colored = np.zeros_like(canvas)
            colored[:, :] = color
            alpha = (mask > 0).astype(np.uint8)[:, :, None]
            canvas = np.where(alpha == 1, cv2.addWeighted(canvas, 0.6, colored, 0.4, 0), canvas)
    return canvas


def draw_combined(scene: SceneFrame) -> np.ndarray:
    """Рисует объединенную визуализацию."""
    det = draw_detection(scene)
    pose = draw_pose(scene)
    parsing = draw_parsing(scene)
    combined = cv2.addWeighted(det, 0.4, pose, 0.3, 0)
    combined = cv2.addWeighted(combined, 0.7, parsing, 0.3, 0)
    return combined
