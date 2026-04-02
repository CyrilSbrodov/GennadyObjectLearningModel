from __future__ import annotations

from typing import Final

from src.models.schemas import PoseResult
from src.representation.schemas import Keypoint2D

# Порядок индексов соответствует MediaPipe PoseLandmark.
MEDIAPIPE_KEYPOINT_NAMES: Final[list[str]] = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]


def pose_to_keypoint_dict(pose: PoseResult | None) -> dict[str, Keypoint2D]:
    """Преобразует PoseResult в словарь именованных ключевых точек."""
    if pose is None:
        return {}
    result: dict[str, Keypoint2D] = {}
    for index, point in enumerate(pose.keypoints):
        name = MEDIAPIPE_KEYPOINT_NAMES[index] if index < len(MEDIAPIPE_KEYPOINT_NAMES) else f"kp_{index}"
        result[name] = Keypoint2D(x=float(point.x), y=float(point.y), confidence=float(point.visibility))
    return result


def bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    """Возвращает центр bbox."""
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0
