from __future__ import annotations

import importlib
import importlib.util
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch


@dataclass(slots=True)
class SAM2Prediction:
    """Результат SAM2 для одного prompt-объекта."""

    mask: np.ndarray
    score: float
    logits: np.ndarray | None


class SAM2Adapter:
    """Низкоуровневый адаптер SAM2: загрузка и предикт масок."""

    def __init__(self, checkpoint_path: str, model_cfg: str, device: str) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.checkpoint_path = Path(checkpoint_path)
        self.model_cfg = model_cfg
        self.device = self._resolve_device(device)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"SAM2 checkpoint не найден: {self.checkpoint_path}")

        self.resolved_model_cfg = self._resolve_model_cfg(model_cfg)
        self._predictor = self._build_predictor()
        self.logger.info(
            "SAM2 инициализирован: cfg=%s, checkpoint=%s, device=%s",
            self.resolved_model_cfg,
            str(self.checkpoint_path),
            self.device,
        )

    def _build_predictor(self):
        if importlib.util.find_spec("sam2") is None:
            raise ModuleNotFoundError(
                "Пакет `sam2` недоступен. Установите его перед запуском backend sam2."
            )

        build_mod = importlib.import_module("sam2.build_sam")
        predictor_mod = importlib.import_module("sam2.sam2_image_predictor")

        build_sam2 = getattr(build_mod, "build_sam2")
        predictor_cls = getattr(predictor_mod, "SAM2ImagePredictor")

        try:
            model = build_sam2(self.resolved_model_cfg, str(self.checkpoint_path), device=self.device)
        except Exception as exc:
            raise RuntimeError(
                "Не удалось загрузить SAM2 модель. Проверьте --sam2-config "
                f"({self.resolved_model_cfg}) и checkpoint ({self.checkpoint_path})."
            ) from exc
        return predictor_cls(model)

    def _resolve_model_cfg(self, model_cfg: str) -> str:
        """Мягко нормализует SAM2 config: существующий путь или yaml-идентификатор для build_sam2."""
        cfg_candidate = str(model_cfg).strip()
        if not cfg_candidate:
            raise ValueError("SAM2 config пустой. Передайте --sam2-config.")

        cfg_path = Path(cfg_candidate)
        if cfg_path.exists():
            resolved = str(cfg_path.resolve())
            self.logger.info("SAM2 config распознан как путь: %s", resolved)
            return resolved

        if not cfg_candidate.endswith(".yaml"):
            raise ValueError(
                "SAM2 config должен быть либо существующим путем к .yaml, "
                "либо yaml-идентификатором, который сможет разрешить sam2/hydra "
                "на этапе build_sam2 (например, configs/sam2.1/sam2.1_hiera_l.yaml)."
            )

        self.logger.info("SAM2 config передан как yaml-идентификатор: %s", cfg_candidate)
        return cfg_candidate

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def predict_masks(
        self,
        frame: np.ndarray,
        boxes: list[tuple[int, int, int, int]],
        points: list[np.ndarray | None] | None = None,
        point_labels: list[np.ndarray | None] | None = None,
    ) -> tuple[list[SAM2Prediction | None], float]:
        """Запускает SAM2 по списку box prompts и опциональным точкам."""
        if not boxes:
            return [], 0.0

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._predictor.set_image(rgb)

        started = time.perf_counter()
        predictions: list[SAM2Prediction | None] = []

        for idx, box in enumerate(boxes):
            box_xyxy = np.array(box, dtype=np.float32)
            prompt_points = points[idx] if points and idx < len(points) else None
            prompt_labels = point_labels[idx] if point_labels and idx < len(point_labels) else None

            kwargs = {
                "box": box_xyxy,
                "multimask_output": True,
            }
            if prompt_points is not None and prompt_points.size > 0:
                kwargs["point_coords"] = prompt_points.astype(np.float32)
                kwargs["point_labels"] = prompt_labels.astype(np.int32) if prompt_labels is not None else np.ones(
                    (prompt_points.shape[0],), dtype=np.int32
                )

            masks, scores, logits = self._predictor.predict(**kwargs)
            if masks is None or len(masks) == 0:
                predictions.append(None)
                continue

            scores_arr = np.asarray(scores, dtype=np.float32)
            best_idx = int(np.argmax(scores_arr))
            best_mask = (np.asarray(masks[best_idx]) > 0).astype(np.uint8)
            best_logits = None if logits is None else np.asarray(logits[best_idx])
            predictions.append(
                SAM2Prediction(
                    mask=best_mask,
                    score=float(scores_arr[best_idx]),
                    logits=best_logits,
                )
            )

        elapsed = time.perf_counter() - started
        return predictions, elapsed
