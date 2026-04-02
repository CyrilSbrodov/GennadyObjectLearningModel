from __future__ import annotations

import logging

import numpy as np
import torch
from ultralytics import YOLO

from src.interfaces.contracts import Detector
from src.models.schemas import Detection


class YOLODetector(Detector):
    """Детектор людей на базе YOLO из ultralytics."""

    def __init__(self, model_name: str = "yolov8n.pt", device: str = "cpu") -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = YOLO(model_name)
        self.device = self._resolve_device(device)
        # Явно переносим модель на CUDA, чтобы детекция не откатывалась на CPU незаметно.
        self.model.to(self.device)
        self.logger.info("YOLO инициализирован на устройстве: %s", self.device)

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Нормализует устройство и бережно откатывается на CPU, если CUDA недоступна."""
        if device == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Выполняет детекцию людей и возвращает канонические боксы."""
        results = self.model.predict(source=frame, device=self.device, verbose=False)
        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls = int(box.cls.item())
                if cls != 0:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    Detection(
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        confidence=float(box.conf.item()),
                    )
                )
        return detections
