from __future__ import annotations

import numpy as np

from src.interfaces.contracts import Renderer
from src.models.schemas import SceneFrame
from src.visualization.drawing import draw_combined, draw_detection, draw_parsing, draw_pose


class OpenCVRenderer(Renderer):
    """Рендерер отладочных изображений на OpenCV."""

    def render(self, scene: SceneFrame) -> dict[str, np.ndarray]:
        """Формирует полный набор отладочных изображений."""
        skeleton = draw_pose(scene)
        parsing = draw_parsing(scene)
        detection = draw_detection(scene)
        combined = draw_combined(scene)
        return {
            "skeleton": skeleton,
            "parsing": parsing,
            "detection": detection,
            "combined": combined,
        }
