from __future__ import annotations

import numpy as np

from src.interfaces.contracts import PoseExtractor
from src.models.schemas import Detection, PoseKeypoint, PoseResult


class MockPoseExtractor(PoseExtractor):
    """Детерминированный мок позы."""

    def extract(self, frame: np.ndarray, detections: list[Detection]) -> list[PoseResult]:
        """Генерирует несколько опорных точек в пределах бокса."""
        results: list[PoseResult] = []
        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det.bbox
            points = [
                PoseKeypoint(x=(x1 + x2) / 2, y=y1 + 10, visibility=1.0),
                PoseKeypoint(x=x1 + 20, y=(y1 + y2) / 2, visibility=1.0),
                PoseKeypoint(x=x2 - 20, y=(y1 + y2) / 2, visibility=1.0),
            ]
            results.append(PoseResult(detection_idx=idx, keypoints=points))
        return results
