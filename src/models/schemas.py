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
