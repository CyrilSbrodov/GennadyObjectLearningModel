from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from src.interfaces.contracts import Detector, HumanParser, PoseExtractor, Renderer, SceneBuilder, Tracker
from src.io.output_writer import OutputWriter


class PipelineOrchestrator:
    """Оркестратор конвейера: detection, pose, parsing, tracking, rendering."""

    def __init__(
        self,
        detector: Detector,
        pose_extractor: PoseExtractor,
        parser: HumanParser,
        tracker: Tracker,
        scene_builder: SceneBuilder,
        renderer: Renderer,
        writer: OutputWriter,
        parsing_interval: int,
    ) -> None:
        self.detector = detector
        self.pose_extractor = pose_extractor
        self.parser = parser
        self.tracker = tracker
        self.scene_builder = scene_builder
        self.renderer = renderer
        self.writer = writer
        self.parsing_interval = max(1, parsing_interval)
        self.logger = logging.getLogger(self.__class__.__name__)

    def process_image(self, image: np.ndarray, base_name: str) -> None:
        """Обрабатывает одно изображение как один кадр."""
        self._process_frame(image, base_name=base_name, frame_idx=0)

    def process_video(self, video_path: str, base_name: str) -> None:
        """Обрабатывает видео покадрово."""
        capture = cv2.VideoCapture(video_path)
        frame_idx = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            self._process_frame(frame, base_name=base_name, frame_idx=frame_idx)
            frame_idx += 1
        capture.release()

    def _process_frame(self, frame: np.ndarray, base_name: str, frame_idx: int) -> None:
        """Применяет все модули к одному кадру."""
        started_at = time.perf_counter()
        detections = self.detector.detect(frame)
        poses = self.pose_extractor.extract(frame, detections)
        parse_enabled = frame_idx % self.parsing_interval == 0
        if parse_enabled:
            self.logger.info("Запуск парсинга для кадра %d", frame_idx)
        parsed = self.parser.parse(frame, detections) if parse_enabled else None
        tracked = self.tracker.update(detections, poses, parsed)
        scene = self.scene_builder.build(frame, detections, poses, tracked)
        scene.frame_index = frame_idx
        images = self.renderer.render(scene)
        self.writer.save(base_name=base_name, frame_idx=frame_idx, images=images)
        elapsed = time.perf_counter() - started_at
        fps = 1.0 / elapsed if elapsed > 0 else 0.0
        self.logger.info(
            "Кадр %d обработан: detections=%d, parsing=%s, fps=%.2f",
            frame_idx,
            len(detections),
            "on" if parse_enabled else "off",
            fps,
        )
