from __future__ import annotations

import numpy as np

from src.interfaces.contracts import SceneBuilder
from src.models.schemas import Detection, PoseResult, SceneFrame, TrackedHuman
from src.representation.builder import HumanRepresentationBuilder


class BasicSceneBuilder(SceneBuilder):
    """Базовая сборка сцены из результатов модулей."""

    def __init__(self, representation_builder: HumanRepresentationBuilder | None = None) -> None:
        self.representation_builder = representation_builder or HumanRepresentationBuilder()

    def build(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        poses: list[PoseResult],
        tracked: list[TrackedHuman],
        frame_index: int = 0,
    ) -> SceneFrame:
        """Возвращает объект сцены для текущего кадра."""
        parsing_by_detection = {
            item.detection_idx: item
            for item in (track.parsed for track in tracked)
            if item is not None
        }
        representations = [
            self.representation_builder.build_for_tracked_human(
                tracked_human=item,
                frame_index=frame_index,
                frame_shape=frame.shape[:2],
                frame=frame,
            )
            for item in tracked
        ]
        return SceneFrame(
            frame_index=frame_index,
            frame=frame,
            detections=detections,
            poses=poses,
            tracked=tracked,
            parsing_by_detection=parsing_by_detection,
            human_representations=representations,
        )
