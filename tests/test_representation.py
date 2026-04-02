from __future__ import annotations

import unittest

import numpy as np

from src.implementations.rendering.opencv_renderer import OpenCVRenderer
from src.implementations.scene.basic_scene_builder import BasicSceneBuilder
from src.models.schemas import Detection, ParsedHuman, PoseKeypoint, PoseResult, SceneFrame, TrackedHuman
from src.representation.builder import HumanRepresentationBuilder
from src.representation.state_rules import build_person_mask


class RepresentationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = np.zeros((120, 160, 3), dtype=np.uint8)
        self.det = Detection(bbox=(20, 10, 100, 110), confidence=0.9)

    def test_builder_without_parsing(self) -> None:
        tracked = TrackedHuman(track_id=1, detection=self.det, pose=None, parsed=None)
        representation = HumanRepresentationBuilder().build_for_tracked_human(tracked)
        self.assertEqual(representation.human_id, "human_1")
        self.assertEqual(representation.bbox, self.det.bbox)
        self.assertIsNone(representation.person_mask)

    def test_builder_without_pose(self) -> None:
        parsing = ParsedHuman(
            detection_idx=0,
            masks={"upper_clothes": np.ones((120, 160), dtype=np.uint8)},
            confidence=0.8,
            model_version="test",
        )
        tracked = TrackedHuman(track_id=2, detection=self.det, pose=None, parsed=parsing)
        representation = HumanRepresentationBuilder().build_for_tracked_human(tracked)
        self.assertEqual(representation.state.pose_state, "unknown_pose")
        self.assertGreaterEqual(len(representation.garments), 1)

    def test_person_mask_union(self) -> None:
        m1 = np.zeros((10, 10), dtype=np.uint8)
        m2 = np.zeros((10, 10), dtype=np.uint8)
        m1[1:3, 1:3] = 1
        m2[5:7, 5:7] = 1
        region = build_person_mask({"a": m1, "b": m2}, confidence=1.0)
        self.assertIsNotNone(region)
        assert region is not None
        self.assertEqual(int(np.count_nonzero(region.mask)), 8)

    def test_overlay_render_shape(self) -> None:
        pose = PoseResult(
            detection_idx=0,
            keypoints=[PoseKeypoint(x=40, y=40, visibility=0.9), PoseKeypoint(x=80, y=40, visibility=0.9)],
        )
        tracked = TrackedHuman(track_id=3, detection=self.det, pose=pose, parsed=None)
        scene = BasicSceneBuilder().build(self.frame, [self.det], [pose], [tracked])
        images = OpenCVRenderer().render(scene)
        self.assertIn("representation_overlay", images)
        self.assertIn("representation_masks", images)
        self.assertEqual(images["representation_overlay"].shape, self.frame.shape)

    def test_summary_panel_render(self) -> None:
        scene = SceneFrame(frame_index=0, frame=self.frame, detections=[], poses=[], tracked=[])
        images = OpenCVRenderer().render(scene)
        self.assertIn("summary_panel", images)
        self.assertEqual(images["summary_panel"].ndim, 3)


if __name__ == "__main__":
    unittest.main()
