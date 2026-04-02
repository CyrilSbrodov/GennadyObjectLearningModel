from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.representation.schemas import HumanRepresentation

PARSING_LABELS: list[str] = [
    "face",
    "hair",
    "upper_clothes",
    "lower_clothes",
    "left_hand",
    "right_hand",
    "left_leg",
    "right_leg",
    "shoes",
]

PARSING_LABELS_V2: list[str] = [
    "head",
    "face",
    "hair",
    "neck",
    "chest_left",
    "chest_right",
    "abdomen",
    "pelvis",
    "glute_left",
    "glute_right",
    "shoulder_left",
    "shoulder_right",
    "upper_arm_left",
    "upper_arm_right",
    "forearm_left",
    "forearm_right",
    "hand_left",
    "hand_right",
    "thigh_left",
    "thigh_right",
    "knee_left",
    "knee_right",
    "calf_left",
    "calf_right",
    "foot_left",
    "foot_right",
    "back_upper",
    "back_lower",
    "breast_areola",
]

PARSING_LABELS_SAM2: list[str] = [
    "person_mask",
    "head",
    "torso",
    "left_arm",
    "right_arm",
    "left_leg",
    "right_leg",
]


@dataclass(slots=True)
class Detection:
    """Детекция человека."""

    bbox: tuple[int, int, int, int]
    confidence: float


@dataclass(slots=True)
class PoseKeypoint:
    """Ключевая точка позы."""

    x: float
    y: float
    visibility: float


@dataclass(slots=True)
class PoseResult:
    """Поза, привязанная к индексу детекции."""

    detection_idx: int
    keypoints: list[PoseKeypoint]


@dataclass(slots=True)
class ParsedHuman:
    """Результат парсинга для человека."""

    detection_idx: int
    masks: dict[str, np.ndarray]
    confidence: float
    model_version: str
    schema_version: str = "v1"
    label_confidence: dict[str, float] = field(default_factory=dict)
    debug: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class TrackedHuman:
    """Состояние человека в треке."""

    track_id: int
    detection: Detection
    pose: PoseResult | None
    parsed: ParsedHuman | None


@dataclass(slots=True)
class SceneFrame:
    """Полная сцена кадра для визуализации."""

    frame_index: int
    frame: np.ndarray
    detections: list[Detection] = field(default_factory=list)
    poses: list[PoseResult] = field(default_factory=list)
    tracked: list[TrackedHuman] = field(default_factory=list)
    parsing_by_detection: dict[int, ParsedHuman] = field(default_factory=dict)
    human_representations: list["HumanRepresentation"] = field(default_factory=list)
