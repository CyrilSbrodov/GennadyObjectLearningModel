from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
import logging

from src.interfaces.contracts import Tracker
from src.models.schemas import Detection, ParsedHuman, PoseResult, TrackedHuman


@dataclass(slots=True)
class _TrackState:
    track_id: int
    bbox: tuple[int, int, int, int]
    parsed: ParsedHuman | None
    history_confidence: deque[float]


class SimpleTracker(Tracker):
    """Простой трекер с переносом масок по сдвигу рамки."""

    def __init__(self, iou_threshold: float = 0.3, history_size: int = 8, confidence_decay: float = 0.9) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.iou_threshold = iou_threshold
        self.history_size = max(1, history_size)
        self.confidence_decay = float(np.clip(confidence_decay, 0.0, 1.0))
        self._next_id = 1
        self._states: list[_TrackState] = []

    def update(
        self,
        detections: list[Detection],
        poses: list[PoseResult],
        parsed: list[ParsedHuman] | None,
    ) -> list[TrackedHuman]:
        """Сопоставляет детекции с треками и заполняет пропуски парсинга."""
        parsed_by_idx = {item.detection_idx: item for item in (parsed or [])}
        pose_by_idx = {item.detection_idx: item for item in poses}
        matched_tracks: list[TrackedHuman] = []
        new_states: list[_TrackState] = []

        for idx, det in enumerate(detections):
            best_state, best_iou = self._match_state(det.bbox)
            current_parsed = parsed_by_idx.get(idx)
            if best_state is None or best_iou < self.iou_threshold:
                track_id = self._next_id
                self._next_id += 1
                state = _TrackState(
                    track_id=track_id,
                    bbox=det.bbox,
                    parsed=current_parsed,
                    history_confidence=deque([current_parsed.confidence] if current_parsed else [], maxlen=self.history_size),
                )
            else:
                propagated = self._propagate_mask(best_state.parsed, best_state.bbox, det.bbox)
                fused = self._fuse_parsed(current_parsed, propagated)
                history = deque(best_state.history_confidence, maxlen=self.history_size)
                if fused is not None:
                    history.append(fused.confidence)
                state = _TrackState(track_id=best_state.track_id, bbox=det.bbox, parsed=fused, history_confidence=history)
            new_states.append(state)
            matched_tracks.append(
                TrackedHuman(
                    track_id=state.track_id,
                    detection=det,
                    pose=pose_by_idx.get(idx),
                    parsed=state.parsed,
                )
            )

        self._states = new_states
        return matched_tracks

    def _match_state(self, bbox: tuple[int, int, int, int]) -> tuple[_TrackState | None, float]:
        """Находит лучший предыдущий трек по IoU."""
        best_state: _TrackState | None = None
        best_iou = 0.0
        for state in self._states:
            iou = self._iou(state.bbox, bbox)
            if iou > best_iou:
                best_iou = iou
                best_state = state
        return best_state, best_iou

    @staticmethod
    def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        """Считает IoU двух боксов."""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter == 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / max(1, area_a + area_b - inter)

    @staticmethod
    def _propagate_mask(
        parsed: ParsedHuman | None,
        prev_bbox: tuple[int, int, int, int],
        curr_bbox: tuple[int, int, int, int],
    ) -> ParsedHuman | None:
        """Переносит маски по сдвигу между рамками."""
        if parsed is None:
            return None
        dx = curr_bbox[0] - prev_bbox[0]
        dy = curr_bbox[1] - prev_bbox[1]
        new_masks: dict[str, np.ndarray] = {}
        for label, mask in parsed.masks.items():
            h, w = mask.shape[:2]
            transform = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted = cv2.warpAffine(mask, transform, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            new_masks[label] = shifted
        return ParsedHuman(
            detection_idx=parsed.detection_idx,
            masks=new_masks,
            confidence=parsed.confidence,
            model_version=parsed.model_version,
            schema_version=parsed.schema_version,
            label_confidence={k: v for k, v in parsed.label_confidence.items()},
            debug={k: v for k, v in parsed.debug.items()},
        )

    def _fuse_parsed(self, parsed_new: ParsedHuman | None, parsed_propagated: ParsedHuman | None) -> ParsedHuman | None:
        """Confidence-aware blending между новым и протащенным парсингом."""
        if parsed_new is None:
            if parsed_propagated is None:
                return None
            decayed_conf = parsed_propagated.confidence * self.confidence_decay
            propagated_masks = {label: mask.copy() for label, mask in parsed_propagated.masks.items()}
            return ParsedHuman(
                detection_idx=parsed_propagated.detection_idx,
                masks=propagated_masks,
                confidence=decayed_conf,
                model_version=parsed_propagated.model_version,
                schema_version=parsed_propagated.schema_version,
                label_confidence={k: v * self.confidence_decay for k, v in parsed_propagated.label_confidence.items()},
                debug={k: v for k, v in parsed_propagated.debug.items()},
            )
        if parsed_propagated is None:
            return parsed_new
        if parsed_new.schema_version != parsed_propagated.schema_version:
            # Для несовместимых label-space выбираем свежий парсинг без агрессивного смешивания.
            return parsed_new

        fused_masks: dict[str, np.ndarray] = {}
        labels = set(parsed_new.masks) | set(parsed_propagated.masks)

        for label in labels:
            new_mask = parsed_new.masks.get(label)
            old_mask = parsed_propagated.masks.get(label)
            if new_mask is None:
                fused_masks[label] = old_mask.copy()
                continue
            if old_mask is None:
                fused_masks[label] = new_mask.copy()
                continue
            if new_mask.shape != old_mask.shape:
                self.logger.warning(
                    "Skip merge for label=%s because shape mismatch: new=%s old=%s schema=%s",
                    label,
                    new_mask.shape,
                    old_mask.shape,
                    parsed_new.schema_version,
                )
                fused_masks[label] = new_mask.copy()
                continue

            new_weight = parsed_new.label_confidence.get(label, parsed_new.confidence)
            old_weight = parsed_propagated.label_confidence.get(label, parsed_propagated.confidence) * self.confidence_decay
            if new_weight >= old_weight:
                fused_masks[label] = np.where(new_mask > 0, 1, old_mask).astype(np.uint8)
            else:
                fused_masks[label] = np.where(old_mask > 0, 1, new_mask).astype(np.uint8)

        total_conf = max(parsed_new.confidence, parsed_propagated.confidence * self.confidence_decay)
        merged_label_conf = {**parsed_propagated.label_confidence, **parsed_new.label_confidence}
        return ParsedHuman(
            detection_idx=parsed_new.detection_idx,
            masks=fused_masks,
            confidence=total_conf,
            model_version=parsed_new.model_version,
            schema_version=parsed_new.schema_version,
            label_confidence=merged_label_conf,
            debug={**parsed_propagated.debug, **parsed_new.debug},
        )
