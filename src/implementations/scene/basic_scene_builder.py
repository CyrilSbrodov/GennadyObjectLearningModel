from __future__ import annotations

import numpy as np

from src.interfaces.contracts import SceneBuilder
from src.models.schemas import Detection, PoseResult, SceneFrame, TrackedHuman


class BasicSceneBuilder(SceneBuilder):
    """Базовая сборка сцены из результатов модулей."""

    def build(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        poses: list[PoseResult],
        tracked: list[TrackedHuman],
    ) -> SceneFrame:
        """Возвращает объект сцены для текущего кадра."""
        parsing_by_detection = {
            item.detection_idx: item
            for item in (track.parsed for track in tracked)
            if item is not None
        }
        return SceneFrame(
            frame_index=0,
            frame=frame,
            detections=detections,
            poses=poses,
            tracked=tracked,
            parsing_by_detection=parsing_by_detection,
        )
