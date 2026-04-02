from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2

from src.config.settings import AppConfig
from src.implementations.detection.mock_detector import MockDetector
from src.implementations.detection.yolo_detector import YOLODetector
from src.implementations.parsing.mock_parser import MockParser
from src.implementations.parsing.sam2_anatomy_parser import SAM2AnatomyParser
from src.implementations.parsing.segformer_parser import SegFormerParser
from src.implementations.pose.mediapipe_holistic_adapter import MediapipeHolisticAdapter
from src.implementations.pose.mock_pose import MockPoseExtractor
from src.implementations.rendering.opencv_renderer import OpenCVRenderer
from src.implementations.scene.basic_scene_builder import BasicSceneBuilder
from src.implementations.tracking.simple_tracker import SimpleTracker
from src.io.input_loader import collect_inputs
from src.io.output_writer import OutputWriter
from src.orchestrator.pipeline import FastPipeline, PipelineOrchestrator, SegmentationWorker


def parse_args() -> argparse.Namespace:
    """Парсит аргументы командной строки."""
    parser = argparse.ArgumentParser(description="Модульный CV-конвейер")
    parser.add_argument("--parsing-interval", type=int, default=5)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--use-mock", action="store_true")
    parser.add_argument("--parser-backend", type=str, choices=["segformer", "sam2"], default="segformer")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> AppConfig:
    """Создает конфигурацию приложения из аргументов."""
    root = Path(__file__).resolve().parent
    return AppConfig(
        parsing_interval=args.parsing_interval,
        device=args.device,
        use_mock=args.use_mock,
        parser_backend=args.parser_backend,
        input_photo_dir=root / "input" / "photo",
        input_video_dir=root / "input" / "video",
        output_root=root / "output",
    )


def main() -> None:
    """Точка входа: выбирает входы и запускает конвейер."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = parse_args()
    config = build_config(args)

    if config.use_mock:
        detector = MockDetector()
        pose_extractor = MockPoseExtractor()
        parser = MockParser()
    else:
        detector = YOLODetector(device=config.device)
        pose_extractor = MediapipeHolisticAdapter()
        parser = SAM2AnatomyParser() if config.parser_backend == "sam2" else SegFormerParser(device=config.device)

    orchestrator = PipelineOrchestrator(
        fast_pipeline=FastPipeline(detector=detector, pose_extractor=pose_extractor),
        segmentation_worker=SegmentationWorker(parser=parser),
        tracker=SimpleTracker(),
        scene_builder=BasicSceneBuilder(),
        renderer=OpenCVRenderer(),
        writer=OutputWriter(config.output_root),
        parsing_interval=config.parsing_interval,
    )

    try:
        photos, videos = collect_inputs(config.input_photo_dir, config.input_video_dir)
        for photo in photos:
            image = cv2.imread(str(photo))
            if image is None:
                continue
            orchestrator.process_image(image, photo.stem)

        for video in videos:
            orchestrator.process_video(str(video), video.stem)
    finally:
        # Явно останавливаем воркер, чтобы не терять хвост очереди при завершении.
        orchestrator.close()


if __name__ == "__main__":
    main()
