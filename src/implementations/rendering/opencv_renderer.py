from __future__ import annotations

import numpy as np

from src.interfaces.contracts import Renderer
from src.models.schemas import SceneFrame
from src.visualization.drawing import draw_anatomy_raw_overlay, draw_combined, draw_detection, draw_parsing, draw_pose
from src.visualization.representation_drawing import (
    draw_representation_debug,
    draw_representation_masks_garments,
    draw_representation_masks,
    draw_representation_masks_normalized,
    draw_representation_masks_raw,
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
        anatomy_raw_overlay = draw_anatomy_raw_overlay(scene)
        representation_overlay = draw_representation_overlay(scene)
        representation_debug = draw_representation_debug(scene)
        representation_masks = draw_representation_masks(scene)
        representation_masks_raw = draw_representation_masks_raw(scene)
        representation_masks_normalized = draw_representation_masks_normalized(scene)
        representation_masks_garments = draw_representation_masks_garments(scene)
        images = {
            "skeleton": skeleton,
            "parsing": parsing,
            "detection": detection,
            "combined": combined,
            "anatomy_raw_overlay": anatomy_raw_overlay,
            "representation_overlay": representation_overlay,
            "representation_debug": representation_debug,
            "representation_masks": representation_masks,
            "representation_masks_raw": representation_masks_raw,
            "representation_masks_normalized": representation_masks_normalized,
            "representation_masks_garments": representation_masks_garments,
        }
        images["summary_panel"] = draw_summary_panel(images)
        return images
