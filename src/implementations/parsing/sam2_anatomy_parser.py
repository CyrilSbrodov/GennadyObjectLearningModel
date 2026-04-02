from __future__ import annotations

import numpy as np

from src.interfaces.contracts import HumanParser
from src.models.schemas import Detection, PARSING_LABELS_V2, ParsedHuman


class SAM2AnatomyParser(HumanParser):
    """Заглушка анатомического парсера v2 (SAM2 + pose prompts).

    Текущая версия повторяет простую геометрию по bbox и хранит структуру,
    необходимую для дальнейшей интеграции prompt-based сегментации.
    """

    MODEL_VERSION = "sam2-anatomy-stub-v0"

    def parse(self, frame: np.ndarray, detections: list[Detection]) -> list[ParsedHuman]:
        h, w = frame.shape[:2]
        parsed: list[ParsedHuman] = []

        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det.bbox
            masks = {label: np.zeros((h, w), dtype=np.uint8) for label in PARSING_LABELS_V2}
            label_confidence = {label: 0.0 for label in PARSING_LABELS_V2}

            if x2 > x1 and y2 > y1:
                width = x2 - x1
                height = y2 - y1
                y_head_end = y1 + max(1, height // 6)
                y_chest_end = y1 + max(1, height // 3)
                y_abdomen_end = y1 + max(1, height // 2)
                y_pelvis_end = y1 + max(1, (2 * height) // 3)

                left_mid = x1 + width // 2

                masks["head"][y1:y_head_end, x1:x2] = 1
                masks["face"][y1:y_head_end, x1 + width // 4 : x2 - width // 4] = 1
                masks["chest_left"][y_head_end:y_chest_end, x1:left_mid] = 1
                masks["chest_right"][y_head_end:y_chest_end, left_mid:x2] = 1
                masks["abdomen"][y_chest_end:y_abdomen_end, x1:x2] = 1
                masks["pelvis"][y_abdomen_end:y_pelvis_end, x1:x2] = 1
                masks["thigh_left"][y_pelvis_end:y2, x1:left_mid] = 1
                masks["thigh_right"][y_pelvis_end:y2, left_mid:x2] = 1

                for label in ["head", "face", "chest_left", "chest_right", "abdomen", "pelvis", "thigh_left", "thigh_right"]:
                    label_confidence[label] = 0.65

            parsed.append(
                ParsedHuman(
                    detection_idx=idx,
                    masks=masks,
                    confidence=0.65,
                    model_version=self.MODEL_VERSION,
                    schema_version="v2",
                    label_confidence=label_confidence,
                )
            )

        return parsed
