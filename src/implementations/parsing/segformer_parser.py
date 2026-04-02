from __future__ import annotations

import cv2
import numpy as np

from src.implementations.parsing.segformer_adapter import MODEL_NAME, SegFormerAdapter
from src.interfaces.contracts import HumanParser
from src.models.schemas import Detection, PARSING_LABELS, ParsedHuman


class SegFormerParser(HumanParser):
    """Парсер одежды на базе SegFormer с маппингом к каноническим меткам."""

    def __init__(self, device: str = "cpu", parsing_size: int = 256) -> None:
        self.adapter = SegFormerAdapter(device=device, parsing_size=parsing_size)
        self.label_id_to_name = self._build_label_map()
        self.canonical_map = {
            "face": {"face"},
            "hair": {"hair"},
            "upper_clothes": {"upper-clothes", "upper_clothes", "coat", "dress", "scarf"},
            "lower_clothes": {"pants", "skirt", "lower-clothes", "lower_clothes"},
            "left_hand": {"left-arm", "left_arm", "left-hand", "left_hand", "glove"},
            "right_hand": {"right-arm", "right_arm", "right-hand", "right_hand", "glove"},
            "left_leg": {"left-leg", "left_leg", "left-shoe", "left_shoe", "socks"},
            "right_leg": {"right-leg", "right_leg", "right-shoe", "right_shoe", "socks"},
            "shoes": {"left-shoe", "left_shoe", "right-shoe", "right_shoe", "shoes"},
        }
        self.canonical_ids = {
            label: np.array(sorted(self._mapped_ids(label)), dtype=np.uint8) for label in PARSING_LABELS
        }

    def parse(self, frame: np.ndarray, detections: list[Detection]) -> list[ParsedHuman]:
        """Запускает SegFormer один раз на весь кадр и режет маски по детекциям."""
        if not detections:
            return []

        pred = self.adapter.infer_frame(frame)
        parsed_list: list[ParsedHuman] = []
        frame_h, frame_w = frame.shape[:2]

        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det.bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_w, x2), min(frame_h, y2)
            if x2 <= x1 or y2 <= y1:
                parsed_list.append(
                    ParsedHuman(
                        detection_idx=idx,
                        masks=self._empty_masks(frame.shape[:2]),
                        confidence=0.0,
                        model_version=MODEL_NAME,
                        schema_version="v1",
                    )
                )
                continue

            roi_pred = pred[y1:y2, x1:x2]
            masks = self._empty_masks(frame.shape[:2])
            human_pixels = int((roi_pred > 0).sum())
            canonical_pixels = 0

            for canonical_label in PARSING_LABELS:
                ids = self.canonical_ids[canonical_label]
                if ids.size == 0:
                    local_mask = np.zeros_like(roi_pred, dtype=np.uint8)
                else:
                    # Векторная операция нужна, чтобы убрать дорогие циклы по каждому пикселю.
                    local_mask = np.isin(roi_pred, ids).astype(np.uint8)
                canonical_pixels += int(local_mask.sum())
                masks[canonical_label][y1:y2, x1:x2] = local_mask

            confidence = float(canonical_pixels / max(1, human_pixels)) if human_pixels > 0 else 0.0
            parsed_list.append(
                ParsedHuman(
                    detection_idx=idx,
                    masks=masks,
                    confidence=confidence,
                    model_version=MODEL_NAME,
                    schema_version="v1",
                )
            )
        return parsed_list

    def _build_label_map(self) -> dict[int, str]:
        """Строит карту id->label из конфигурации модели."""
        raw_map = self.adapter.model.config.id2label
        mapped: dict[int, str] = {}
        for key, value in raw_map.items():
            idx = int(key)
            normalized = str(value).strip().lower().replace(" ", "-")
            mapped[idx] = normalized
        return mapped

    def _mapped_ids(self, canonical_label: str) -> set[int]:
        """Возвращает набор id модели, соответствующий канонической метке."""
        aliases = self.canonical_map[canonical_label]
        result: set[int] = set()
        for idx, name in self.label_id_to_name.items():
            if name in aliases:
                result.add(idx)
        return result

    @staticmethod
    def _empty_masks(shape: tuple[int, int]) -> dict[str, np.ndarray]:
        """Создает пустой набор канонических масок."""
        return {label: np.zeros(shape, dtype=np.uint8) for label in PARSING_LABELS}
