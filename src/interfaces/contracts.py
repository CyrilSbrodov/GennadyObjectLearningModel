from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.models.schemas import Detection, ParsedHuman, PoseResult, SceneFrame, TrackedHuman


class Detector(ABC):
    """Интерфейс детектора людей."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Возвращает список детекций людей на кадре."""


class PoseExtractor(ABC):
    """Интерфейс извлечения позы."""

    @abstractmethod
    def extract(self, frame: np.ndarray, detections: list[Detection]) -> list[PoseResult]:
        """Возвращает позы для найденных людей."""


class HumanParser(ABC):
    """Интерфейс парсинга одежды и частей тела."""

    @abstractmethod
    def parse(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        poses: list[PoseResult] | None = None,
    ) -> list[ParsedHuman]:
        """Возвращает сегментацию человека по каноническим классам."""


class Tracker(ABC):
    """Интерфейс трекинга людей между кадрами."""

    @abstractmethod
    def update(
        self,
        detections: list[Detection],
        poses: list[PoseResult],
        parsed: list[ParsedHuman] | None,
    ) -> list[TrackedHuman]:
        """Обновляет треки и заполняет пропуски по парсингу."""


class SceneBuilder(ABC):
    """Интерфейс построения сцены из модулей."""

    @abstractmethod
    def build(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        poses: list[PoseResult],
        tracked: list[TrackedHuman],
        frame_index: int = 0,
    ) -> SceneFrame:
        """Собирает каноническое представление сцены."""


class Renderer(ABC):
    """Интерфейс отрисовки отладочных визуализаций."""

    @abstractmethod
    def render(self, scene: SceneFrame) -> dict[str, np.ndarray]:
        """Возвращает набор изображений для сохранения."""
