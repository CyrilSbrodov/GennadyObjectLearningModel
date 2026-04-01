from __future__ import annotations

import cv2
import numpy as np

from src.interfaces.contracts import HumanParser
from src.models.schemas import Detection, PARSING_LABELS, ParsedHuman
from src.implementations.parsing.segformer_adapter import MODEL_NAME, SegFormerAdapter


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

    def parse(self, frame: np.ndarray, detections: list[Detection]) -> list[ParsedHuman]:
        """Запускает парсинг внутри ROI человека."""
        parsed_list: list[ParsedHuman] = []
        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det.bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                parsed_list.append(
                    ParsedHuman(
                        detection_idx=idx,
                        masks=self._empty_masks(frame.shape[:2]),
                        confidence=0.0,
                        model_version=MODEL_NAME,
                    )
                )
                continue

            pred = self.adapter.infer_roi(roi)
            pred = cv2.resize(pred, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

            masks = self._empty_masks(frame.shape[:2])
            canonical_pixels = 0
            human_pixels = int((pred > 0).sum())
            for canonical_label in PARSING_LABELS:
                mapped_ids = self._mapped_ids(canonical_label)
                local_mask = np.isin(pred, list(mapped_ids)).astype(np.uint8)
                canonical_pixels += int(local_mask.sum())
                full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                full_mask[y1:y2, x1:x2] = local_mask
                masks[canonical_label] = full_mask
            confidence = float(canonical_pixels / max(1, human_pixels)) if human_pixels > 0 else 0.0
            parsed_list.append(
                ParsedHuman(
                    detection_idx=idx,
                    masks=masks,
                    confidence=confidence,
                    model_version=MODEL_NAME,
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
