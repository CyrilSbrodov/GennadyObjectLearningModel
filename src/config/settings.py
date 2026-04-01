from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AppConfig:
    """Конфигурация запуска конвейера."""

    parsing_interval: int
    device: str
    use_mock: bool
    input_photo_dir: Path
    input_video_dir: Path
    output_root: Path
