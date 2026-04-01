from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np

from src.interfaces.contracts import PoseExtractor
from src.models.schemas import Detection, PoseKeypoint, PoseResult


class MediapipeHolisticAdapter(PoseExtractor):
    """Адаптер позы на базе MediaPipe Holistic."""

    def __init__(self) -> None:
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def extract(self, frame: np.ndarray, detections: list[Detection]) -> list[PoseResult]:
        """Извлекает позу из ROI каждой детекции."""
        poses: list[PoseResult] = []
        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det.bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                poses.append(PoseResult(detection_idx=idx, keypoints=[]))
                continue
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            result = self.holistic.process(rgb)
            keypoints: list[PoseKeypoint] = []
            if result.pose_landmarks is not None:
                for landmark in result.pose_landmarks.landmark:
                    keypoints.append(
                        PoseKeypoint(
                            x=x1 + landmark.x * (x2 - x1),
                            y=y1 + landmark.y * (y2 - y1),
                            visibility=landmark.visibility,
                        )
                    )
            poses.append(PoseResult(detection_idx=idx, keypoints=keypoints))
        return poses
