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
        self.representation_overlay_dir = root / "representation_overlay"
        self.representation_debug_dir = root / "representation_debug"
        self.representation_masks_dir = root / "representation_masks"
        self.representation_masks_raw_dir = root / "representation_masks_raw"
        self.representation_masks_normalized_dir = root / "representation_masks_normalized"
        self.representation_masks_garments_dir = root / "representation_masks_garments"
        self.anatomy_raw_overlay_dir = root / "anatomy_raw_overlay"
        self.summary_panel_dir = root / "summary_panel"
        for directory in [
            self.debug_dir,
            self.skeleton_dir,
            self.parsing_dir,
            self.detection_dir,
            self.combined_dir,
            self.representation_overlay_dir,
            self.representation_debug_dir,
            self.representation_masks_dir,
            self.representation_masks_raw_dir,
            self.representation_masks_normalized_dir,
            self.representation_masks_garments_dir,
            self.anatomy_raw_overlay_dir,
            self.summary_panel_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    def save(self, base_name: str, frame_idx: int, images: dict[str, np.ndarray]) -> None:
        """Сохраняет набор изображений для одного кадра."""
        stem = f"{base_name}_frame_{frame_idx:06d}.png"
        cv2.imwrite(str(self.skeleton_dir / stem), images["skeleton"])
        cv2.imwrite(str(self.parsing_dir / stem), images["parsing"])
        cv2.imwrite(str(self.detection_dir / stem), images["detection"])
        cv2.imwrite(str(self.combined_dir / stem), images["combined"])
        cv2.imwrite(str(self.representation_overlay_dir / stem), images["representation_overlay"])
        cv2.imwrite(str(self.representation_debug_dir / stem), images["representation_debug"])
        cv2.imwrite(str(self.representation_masks_dir / stem), images["representation_masks"])
        cv2.imwrite(str(self.representation_masks_raw_dir / stem), images["representation_masks_raw"])
        cv2.imwrite(str(self.representation_masks_normalized_dir / stem), images["representation_masks_normalized"])
        cv2.imwrite(str(self.representation_masks_garments_dir / stem), images["representation_masks_garments"])
        cv2.imwrite(str(self.anatomy_raw_overlay_dir / stem), images["anatomy_raw_overlay"])
        cv2.imwrite(str(self.summary_panel_dir / stem), images["summary_panel"])
        cv2.imwrite(str(self.debug_dir / stem), images["combined"])
