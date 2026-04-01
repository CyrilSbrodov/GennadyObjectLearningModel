from __future__ import annotations

import numpy as np

from src.interfaces.contracts import Detector
from src.models.schemas import Detection


class MockDetector(Detector):
    """Детерминированный мок детектора для отладки."""

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Возвращает одну фиксированную рамку в центре кадра."""
        h, w = frame.shape[:2]
        x1, y1 = int(w * 0.3), int(h * 0.2)
        x2, y2 = int(w * 0.7), int(h * 0.9)
        return [Detection(bbox=(x1, y1, x2, y2), confidence=0.99)]
