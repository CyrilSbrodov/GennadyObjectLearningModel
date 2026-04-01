from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class OutputWriter:
    """Сохраняет отладочные изображения в целевые директории."""

    def __init__(self, root: Path) -> None:
        self.debug_dir = root / "debug"
        self.skeleton_dir = root / "skeleton"
        self.parsing_dir = root / "parsing"
        self.detection_dir = root / "detection"
        self.combined_dir = root / "combined"
        for directory in [
            self.debug_dir,
            self.skeleton_dir,
            self.parsing_dir,
            self.detection_dir,
            self.combined_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    def save(self, base_name: str, frame_idx: int, images: dict[str, np.ndarray]) -> None:
        """Сохраняет набор изображений для одного кадра."""
        stem = f"{base_name}_frame_{frame_idx:06d}.png"
        cv2.imwrite(str(self.skeleton_dir / stem), images["skeleton"])
        cv2.imwrite(str(self.parsing_dir / stem), images["parsing"])
        cv2.imwrite(str(self.detection_dir / stem), images["detection"])
        cv2.imwrite(str(self.combined_dir / stem), images["combined"])
        cv2.imwrite(str(self.debug_dir / stem), images["combined"])
