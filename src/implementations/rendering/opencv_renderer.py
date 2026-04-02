from __future__ import annotations

import numpy as np

from src.interfaces.contracts import Renderer
from src.models.schemas import SceneFrame
from src.visualization.drawing import draw_combined, draw_detection, draw_parsing, draw_pose
from src.visualization.representation_drawing import (
    draw_representation_debug,
    draw_representation_overlay,
    draw_summary_panel,
)


class OpenCVRenderer(Renderer):
    """Рендерер отладочных изображений на OpenCV."""

    def render(self, scene: SceneFrame) -> dict[str, np.ndarray]:
        """Формирует полный набор отладочных изображений."""
        skeleton = draw_pose(scene)
        parsing = draw_parsing(scene)
        detection = draw_detection(scene)
        combined = draw_combined(scene)
        representation_overlay = draw_representation_overlay(scene)
        representation_debug = draw_representation_debug(scene)
        images = {
            "skeleton": skeleton,
            "parsing": parsing,
            "detection": detection,
            "combined": combined,
            "representation_overlay": representation_overlay,
            "representation_debug": representation_debug,
        }
        images["summary_panel"] = draw_summary_panel(images)
        return images
