from __future__ import annotations

import numpy as np

from src.interfaces.contracts import HumanParser
from src.models.schemas import Detection, PARSING_LABELS, ParsedHuman


class MockParser(HumanParser):
    """Детерминированный мок парсинга с простыми масками."""

    def parse(self, frame: np.ndarray, detections: list[Detection]) -> list[ParsedHuman]:
        """Создает прямоугольные маски по каноническим классам."""
        h, w = frame.shape[:2]
        output: list[ParsedHuman] = []
        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det.bbox
            masks = {label: np.zeros((h, w), dtype=np.uint8) for label in PARSING_LABELS}
            y_mid = (y1 + y2) // 2
            masks["upper_clothes"][y1:y_mid, x1:x2] = 1
            masks["lower_clothes"][y_mid:y2, x1:x2] = 1
            masks["face"][y1 : y1 + max(1, (y2 - y1) // 8), x1 + (x2 - x1) // 3 : x1 + 2 * (x2 - x1) // 3] = 1
            output.append(
                ParsedHuman(
                    detection_idx=idx,
                    masks=masks,
                    confidence=1.0,
                    model_version="mock-parser-v1",
                )
            )
        return output
