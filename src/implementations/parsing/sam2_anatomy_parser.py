from __future__ import annotations

import logging

import numpy as np

from src.implementations.parsing.sam2_adapter import SAM2Adapter
from src.interfaces.contracts import HumanParser
from src.models.schemas import Detection, PARSING_LABELS_SAM2, ParsedHuman, PoseResult


class SAM2AnatomyParser(HumanParser):
    """Реальный SAM2 backend: person mask + coarse части тела (heuristic)."""

    MODEL_VERSION = "sam2-image-predictor-v1"

    def __init__(
        self,
        checkpoint_path: str,
        model_cfg: str,
        device: str = "cpu",
        use_pose_prompts: bool = True,
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.use_pose_prompts = use_pose_prompts
        self.adapter = SAM2Adapter(checkpoint_path=checkpoint_path, model_cfg=model_cfg, device=device)

    def parse(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        poses: list[PoseResult] | None = None,
    ) -> list[ParsedHuman]:
        if not detections:
            return []

        h, w = frame.shape[:2]
        boxes = [self._sanitize_bbox(det.bbox, (h, w)) for det in detections]
        pose_by_det = {item.detection_idx: item for item in (poses or [])}

        prompts_points: list[np.ndarray | None] = []
        prompts_labels: list[np.ndarray | None] = []
        for idx, bbox in enumerate(boxes):
            pose = pose_by_det.get(idx)
            pts, labels = self._build_prompt_points(pose=pose, bbox=bbox, use_pose=self.use_pose_prompts)
            prompts_points.append(pts)
            prompts_labels.append(labels)

        self.logger.info("SAM2 parse start: detections=%d, pose_prompts=%s", len(detections), self.use_pose_prompts)
        predictions, elapsed = self.adapter.predict_masks(
            frame=frame,
            boxes=boxes,
            points=prompts_points,
            point_labels=prompts_labels,
        )

        parsed: list[ParsedHuman] = []
        for idx, det in enumerate(detections):
            empty_masks = self._empty_masks((h, w))
            pred = predictions[idx] if idx < len(predictions) else None
            bbox = boxes[idx]

            if pred is None:
                self.logger.warning("SAM2 не вернул маску для detection_idx=%d bbox=%s", idx, bbox)
                parsed.append(
                    ParsedHuman(
                        detection_idx=idx,
                        masks=empty_masks,
                        confidence=0.0,
                        model_version=self.MODEL_VERSION,
                        schema_version="sam2",
                        label_confidence={label: 0.0 for label in PARSING_LABELS_SAM2},
                        debug={
                            "prompt_box": bbox,
                            "prompt_points": self._points_to_serializable(prompts_points[idx]),
                            "sam2_score": 0.0,
                            "sam2_error": "no_mask",
                            "mask_shape": [],
                        },
                    )
                )
                continue

            person_mask = self._ensure_full_frame_mask(pred.mask, (h, w))
            masks, label_conf = self._build_coarse_masks(person_mask=person_mask, bbox=bbox, prompt_points=prompts_points[idx])
            masks["person_mask"] = person_mask
            label_conf["person_mask"] = float(pred.score)

            parsed.append(
                ParsedHuman(
                    detection_idx=idx,
                    masks=masks,
                    confidence=float(pred.score),
                    model_version=self.MODEL_VERSION,
                    schema_version="sam2",
                    label_confidence=label_conf,
                    debug={
                        "prompt_box": bbox,
                        "prompt_points": self._points_to_serializable(prompts_points[idx]),
                        "sam2_score": float(pred.score),
                        "mask_shape": [int(v) for v in pred.mask.shape[:2]],
                    },
                )
            )
            self.logger.info("SAM2 detection_idx=%d: score=%.4f, pixels=%d", idx, float(pred.score), int(person_mask.sum()))

        self.logger.info("SAM2 parse завершен: people=%d, elapsed=%.3f c", len(detections), elapsed)
        return parsed

    @staticmethod
    def _sanitize_bbox(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
        h, w = shape
        x1, y1, x2, y2 = bbox
        x1 = int(np.clip(x1, 0, w - 1))
        y1 = int(np.clip(y1, 0, h - 1))
        x2 = int(np.clip(x2, x1 + 1, w))
        y2 = int(np.clip(y2, y1 + 1, h))
        return x1, y1, x2, y2

    @staticmethod
    def _empty_masks(shape: tuple[int, int]) -> dict[str, np.ndarray]:
        return {label: np.zeros(shape, dtype=np.uint8) for label in PARSING_LABELS_SAM2}

    @staticmethod
    def _ensure_full_frame_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        """SAM2ImagePredictor.predict возвращает CxHxW, где HxW — размер исходного изображения."""
        h, w = shape
        if mask.shape[:2] == (h, w):
            return (mask > 0).astype(np.uint8)
        raise ValueError(
            "SAM2 вернул маску с неожиданной формой "
            f"{mask.shape[:2]} при размере кадра {(h, w)}. "
            "Ожидается full-image mask от SAM2ImagePredictor.predict."
        )

    def _build_prompt_points(
        self,
        pose: PoseResult | None,
        bbox: tuple[int, int, int, int],
        use_pose: bool,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        x1, y1, x2, y2 = bbox
        center = np.array([[(x1 + x2) / 2.0, (y1 + y2) / 2.0]], dtype=np.float32)
        if not use_pose or pose is None:
            return center, np.ones((1,), dtype=np.int32)

        points = [center[0]]
        labels = [1]
        for kp in pose.keypoints:
            if kp.visibility < 0.35:
                continue
            if not (x1 <= kp.x <= x2 and y1 <= kp.y <= y2):
                continue
            points.append([kp.x, kp.y])
            labels.append(1)

        if not points:
            return None, None
        return np.asarray(points, dtype=np.float32), np.asarray(labels, dtype=np.int32)

    @staticmethod
    def _build_coarse_masks(
        person_mask: np.ndarray,
        bbox: tuple[int, int, int, int],
        prompt_points: np.ndarray | None,
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        x1, y1, x2, y2 = bbox
        h = max(1, y2 - y1)

        ys = np.where(person_mask > 0)[0]
        if ys.size == 0:
            masks = {label: np.zeros_like(person_mask, dtype=np.uint8) for label in PARSING_LABELS_SAM2}
            return masks, {label: 0.0 for label in PARSING_LABELS_SAM2}

        head_y = int(y1 + 0.2 * h)
        torso_y = int(y1 + 0.55 * h)

        grid_y, grid_x = np.indices(person_mask.shape)
        person = person_mask > 0
        left_side = grid_x < (x1 + x2) // 2
        right_side = ~left_side

        head = person & (grid_y <= head_y)
        torso = person & (grid_y > head_y) & (grid_y <= torso_y)
        legs = person & (grid_y > torso_y)

        left_leg = legs & left_side
        right_leg = legs & right_side

        arm_band_low = int(y1 + 0.22 * h)
        arm_band_high = int(y1 + 0.65 * h)
        arm_zone = person & (grid_y >= arm_band_low) & (grid_y <= arm_band_high)
        left_arm = arm_zone & left_side
        right_arm = arm_zone & right_side

        if prompt_points is not None and len(prompt_points) > 0:
            for px, py in prompt_points:
                if px < (x1 + x2) / 2:
                    left_arm[max(0, int(py) - 4) : int(py) + 5, max(0, int(px) - 4) : int(px) + 5] |= True
                else:
                    right_arm[max(0, int(py) - 4) : int(py) + 5, max(0, int(px) - 4) : int(px) + 5] |= True

        masks = {
            "person_mask": person.astype(np.uint8),
            "head": head.astype(np.uint8),
            "torso": torso.astype(np.uint8),
            "left_arm": left_arm.astype(np.uint8),
            "right_arm": right_arm.astype(np.uint8),
            "left_leg": left_leg.astype(np.uint8),
            "right_leg": right_leg.astype(np.uint8),
        }

        conf = {
            "person_mask": 0.95,
            "head": 0.55,
            "torso": 0.6,
            "left_arm": 0.45,
            "right_arm": 0.45,
            "left_leg": 0.6,
            "right_leg": 0.6,
        }
        return masks, conf

    @staticmethod
    def _points_to_serializable(points: np.ndarray | None) -> list[list[float]]:
        if points is None:
            return []
        return [[float(x), float(y)] for x, y in points]
