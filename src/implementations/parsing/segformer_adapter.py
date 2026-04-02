from __future__ import annotations

from typing import Final

import cv2
import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

MODEL_NAME: Final[str] = "sayeed99/segformer_b3_clothes"


class SegFormerAdapter:
    """Адаптер работы с моделью SegFormer и предсказанием карты классов."""

    def __init__(self, device: str, parsing_size: int) -> None:
        self.device = self._resolve_device(device)
        self.parsing_size = parsing_size
        self.processor = SegformerImageProcessor(do_reduce_labels=False)
        self.model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME).to(self.device)
        self.model.eval()

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Выбирает устройство с безопасным откатом на CPU."""
        if device == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def infer_frame(self, frame: np.ndarray) -> np.ndarray:
        """Возвращает карту id классов для всего кадра, чтобы запускать модель только один раз."""
        src_h, src_w = frame.shape[:2]
        resized = cv2.resize(frame, (self.parsing_size, self.parsing_size), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
        pred = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)
        return cv2.resize(pred, (src_w, src_h), interpolation=cv2.INTER_NEAREST)
