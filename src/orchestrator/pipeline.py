from __future__ import annotations

import logging
import queue
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

import cv2
import numpy as np

from src.interfaces.contracts import Detector, HumanParser, PoseExtractor, Renderer, SceneBuilder, Tracker
from src.io.output_writer import OutputWriter
from src.models.schemas import Detection, ParsedHuman

FrameKey = tuple[str, int]


class FastPipeline:
    """Быстрый контур: детекция + поза для каждого кадра."""

    def __init__(self, detector: Detector, pose_extractor: PoseExtractor) -> None:
        self.detector = detector
        self.pose_extractor = pose_extractor

    def run(self, frame: np.ndarray) -> tuple[list[Detection], list]:
        """Выполняет только те этапы, которые должны идти в реальном времени."""
        detections = self.detector.detect(frame)
        poses = self.pose_extractor.extract(frame, detections)
        return detections, poses


@dataclass(slots=True)
class SegmentationTask:
    """Задача для медленного воркера сегментации."""

    key: FrameKey
    frame: np.ndarray
    detections: list[Detection]


class SegmentationWorker:
    """Асинхронный воркер SegFormer, изолирующий тяжелую модель от realtime-контура."""

    def __init__(self, parser: HumanParser, max_queue_size: int = 8, max_cache_size: int = 128) -> None:
        self.parser = parser
        self.logger = logging.getLogger(self.__class__.__name__)
        self._queue: queue.Queue[SegmentationTask] = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="segmentation-worker")
        self._lock = threading.Lock()
        self._frame_cache: OrderedDict[FrameKey, list[ParsedHuman]] = OrderedDict()
        self._pending_keys: set[FrameKey] = set()
        self._max_cache_size = max(1, max_cache_size)

    def start(self) -> None:
        """Запускает поток заранее, чтобы прогрев и инференс не тормозили обработку кадров."""
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self, drain: bool = True) -> None:
        """Останавливает воркер: при drain=True сначала дожидается полного опустошения очереди."""
        if drain:
            self._queue.join()
        self._stop_event.set()
        try:
            self._queue.put_nowait(SegmentationTask(key=("", -1), frame=np.zeros((1, 1, 3), dtype=np.uint8), detections=[]))
        except queue.Full:
            pass
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def submit(self, key: FrameKey, frame: np.ndarray, detections: list[Detection]) -> bool:
        """Добавляет задачу в очередь без блокировки fast-контура."""
        with self._lock:
            if key in self._frame_cache:
                self.logger.info("Медленный контур: cache hit для %s, повторная постановка не нужна", key)
                self._frame_cache.move_to_end(key)
                return False
            if key in self._pending_keys:
                self.logger.info("Медленный контур: %s уже в очереди, повторная постановка пропущена", key)
                return False

        task = SegmentationTask(key=key, frame=frame.copy(), detections=list(detections))
        try:
            self._queue.put_nowait(task)
            with self._lock:
                self._pending_keys.add(key)
            return True
        except queue.Full:
            self.logger.warning("Медленный контур: очередь заполнена, задача %s отброшена", key)
            return False

    def get_cached_frame(self, key: FrameKey) -> list[ParsedHuman] | None:
        """Возвращает кэшированный результат для кадра, чтобы переиспользовать расчет."""
        with self._lock:
            cached = self._frame_cache.get(key)
            if cached is None:
                return None
            self._frame_cache.move_to_end(key)
            self.logger.info("Медленный контур: cache hit для %s", key)
            return list(cached)

    def _run(self) -> None:
        """Крутит цикл воркера отдельно, чтобы тяжелый инференс не блокировал главный поток."""
        while not self._stop_event.is_set():
            task = self._queue.get()
            if task.key[1] < 0:
                self._queue.task_done()
                continue
            started = time.perf_counter()
            parsed = self.parser.parse(task.frame, task.detections)
            elapsed = time.perf_counter() - started
            with self._lock:
                self._pending_keys.discard(task.key)
                self._frame_cache[task.key] = parsed
                self._frame_cache.move_to_end(task.key)
                while len(self._frame_cache) > self._max_cache_size:
                    self._frame_cache.popitem(last=False)
            self.logger.info(
                "Медленный контур: кадр %s сегментирован за %.3f c, людей=%d",
                task.key,
                elapsed,
                len(parsed),
            )
            self._queue.task_done()


class PipelineOrchestrator:
    """Оркестратор: быстрый realtime-контур + асинхронный медленный контур."""

    def __init__(
        self,
        fast_pipeline: FastPipeline,
        segmentation_worker: SegmentationWorker,
        tracker: Tracker,
        scene_builder: SceneBuilder,
        renderer: Renderer,
        writer: OutputWriter,
        parsing_interval: int,
    ) -> None:
        self.fast_pipeline = fast_pipeline
        self.segmentation_worker = segmentation_worker
        self.tracker = tracker
        self.scene_builder = scene_builder
        self.renderer = renderer
        self.writer = writer
        self.parsing_interval = max(1, parsing_interval)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.segmentation_worker.start()

    def process_image(self, image: np.ndarray, base_name: str) -> None:
        """Обрабатывает изображение как единственный кадр."""
        self._process_frame(image, base_name=base_name, frame_idx=0)

    def process_video(self, video_path: str, base_name: str) -> None:
        """Обрабатывает видео покадрово без ожидания медленного контура."""
        capture = cv2.VideoCapture(video_path)
        frame_idx = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            self._process_frame(frame, base_name=base_name, frame_idx=frame_idx)
            frame_idx += 1
        capture.release()

    def close(self) -> None:
        """Закрывает воркер в режиме дренажа, чтобы завершить уже поставленные задачи."""
        self.segmentation_worker.stop(drain=True)

    def _process_frame(self, frame: np.ndarray, base_name: str, frame_idx: int) -> None:
        """Применяет быстрый контур и подмешивает кэш медленного контура при наличии."""
        started_at = time.perf_counter()
        detections, poses = self.fast_pipeline.run(frame)

        frame_key = (base_name, frame_idx)
        parsed = self.segmentation_worker.get_cached_frame(frame_key)
        parse_requested = frame_idx % self.parsing_interval == 0
        if parse_requested:
            enqueued = self.segmentation_worker.submit(key=frame_key, frame=frame, detections=detections)
            if enqueued:
                self.logger.info("Медленный контур: кадр %s отправлен в очередь", frame_key)

        tracked = self.tracker.update(detections, poses, parsed)
        scene = self.scene_builder.build(frame, detections, poses, tracked)
        scene.frame_index = frame_idx
        images = self.renderer.render(scene)
        self.writer.save(base_name=base_name, frame_idx=frame_idx, images=images)

        elapsed = time.perf_counter() - started_at
        fps = 1.0 / elapsed if elapsed > 0 else 0.0
        self.logger.info(
            "Быстрый контур: кадр %d обработан, detections=%d, cached_parsing=%s, fps=%.2f",
            frame_idx,
            len(detections),
            "yes" if parsed is not None else "no",
            fps,
        )
