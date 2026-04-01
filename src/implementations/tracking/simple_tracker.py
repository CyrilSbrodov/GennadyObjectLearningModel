from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.interfaces.contracts import Tracker
from src.models.schemas import Detection, ParsedHuman, PoseResult, TrackedHuman


@dataclass(slots=True)
class _TrackState:
    track_id: int
    bbox: tuple[int, int, int, int]
    parsed: ParsedHuman | None


class SimpleTracker(Tracker):
    """Простой трекер с переносом масок по сдвигу рамки."""

    def __init__(self, iou_threshold: float = 0.3) -> None:
        self.iou_threshold = iou_threshold
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
                state = _TrackState(track_id=track_id, bbox=det.bbox, parsed=current_parsed)
            else:
                propagated = current_parsed or self._propagate_mask(best_state.parsed, best_state.bbox, det.bbox)
                state = _TrackState(track_id=best_state.track_id, bbox=det.bbox, parsed=propagated)
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
        )
