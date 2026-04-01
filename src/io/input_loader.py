from __future__ import annotations

from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def collect_inputs(photo_dir: Path, video_dir: Path) -> tuple[list[Path], list[Path]]:
    """Собирает изображения и видео из входных директорий."""
    photos = sorted([p for p in photo_dir.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS])
    videos = sorted([p for p in video_dir.glob("*") if p.suffix.lower() in VIDEO_EXTENSIONS])
    return photos, videos
