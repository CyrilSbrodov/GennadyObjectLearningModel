from __future__ import annotations

import unittest

import numpy as np

from src.implementations.rendering.opencv_renderer import OpenCVRenderer
from src.implementations.scene.basic_scene_builder import BasicSceneBuilder
from src.implementations.tracking.simple_tracker import SimpleTracker
from src.models.schemas import Detection, ParsedHuman, PoseKeypoint, PoseResult, SceneFrame, TrackedHuman
from src.representation.builder import HumanRepresentationBuilder
from src.representation.parsing_adapters import create_adapter
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
        self.assertIn("representation_masks_raw", images)
        self.assertIn("representation_masks_normalized", images)
        self.assertIn("representation_masks_garments", images)
        self.assertIn("anatomy_raw_overlay", images)
        self.assertIn("sam2_raw_mask", images)
        self.assertIn("sam2_prompt_debug", images)
        self.assertEqual(images["representation_overlay"].shape, self.frame.shape)

    def test_builder_v2_schema_produces_coarse_and_fine_parts(self) -> None:
        masks = {}
        labels = [
            "head",
            "face",
            "hair",
            "neck",
            "chest_left",
            "chest_right",
            "abdomen",
            "pelvis",
            "shoulder_left",
            "shoulder_right",
            "upper_arm_left",
            "upper_arm_right",
            "forearm_left",
            "forearm_right",
            "hand_left",
            "hand_right",
            "thigh_left",
            "thigh_right",
            "knee_left",
            "knee_right",
            "calf_left",
            "calf_right",
            "foot_left",
            "foot_right",
        ]
        for label in labels:
            masks[label] = np.zeros((120, 160), dtype=np.uint8)
        masks["chest_left"][20:35, 20:45] = 1
        masks["chest_right"][20:35, 45:70] = 1
        masks["abdomen"][35:55, 20:70] = 1
        masks["pelvis"][55:72, 20:70] = 1
        masks["thigh_left"][72:102, 20:45] = 1
        masks["thigh_right"][72:102, 45:70] = 1
        masks["foot_left"][102:110, 20:35] = 1
        masks["foot_right"][102:110, 55:70] = 1
        masks["upper_arm_left"][30:52, 10:22] = 1
        masks["upper_arm_right"][30:52, 70:82] = 1
        masks["forearm_left"][52:70, 8:20] = 1
        masks["forearm_right"][52:70, 72:84] = 1
        masks["hand_left"][70:78, 7:18] = 1
        masks["hand_right"][70:78, 74:85] = 1
        masks["head"][10:20, 30:60] = 1
        masks["face"][12:20, 35:55] = 1
        masks["hair"][8:13, 33:57] = 1

        parsing = ParsedHuman(
            detection_idx=0,
            masks=masks,
            confidence=0.7,
            model_version="sam2-anatomy-stub-v0",
            schema_version="v2",
        )
        tracked = TrackedHuman(track_id=4, detection=self.det, pose=None, parsed=parsing)
        representation = HumanRepresentationBuilder().build_for_tracked_human(tracked, frame=self.frame)
        self.assertIn("torso", representation.body_parts)
        self.assertIn("left_upper_arm", representation.body_parts)
        self.assertIn("right_leg", representation.body_parts)
        self.assertIn("head", representation.body_parts)
        self.assertIn("head_core", representation.body_parts)
        self.assertIn("human_4_garment_upper_inner_0", representation.garments)

    def test_create_adapter_unknown_schema_raises(self) -> None:
        with self.assertRaises(ValueError):
            create_adapter("abc")

    def test_create_adapter_sam2_supported(self) -> None:
        adapter = create_adapter("sam2")
        self.assertEqual(adapter.__class__.__name__, "SAM2ParsingAdapter")

    def test_builder_sam2_schema_no_forced_garments(self) -> None:
        masks = {
            "person_mask": np.zeros((120, 160), dtype=np.uint8),
            "head": np.zeros((120, 160), dtype=np.uint8),
            "torso": np.zeros((120, 160), dtype=np.uint8),
            "left_arm": np.zeros((120, 160), dtype=np.uint8),
            "right_arm": np.zeros((120, 160), dtype=np.uint8),
            "left_leg": np.zeros((120, 160), dtype=np.uint8),
            "right_leg": np.zeros((120, 160), dtype=np.uint8),
        }
        masks["person_mask"][10:110, 20:100] = 1
        masks["torso"][25:70, 30:90] = 1
        masks["left_leg"][70:110, 30:60] = 1
        masks["right_leg"][70:110, 60:90] = 1
        masks["head"][10:25, 45:75] = 1
        parsing = ParsedHuman(
            detection_idx=0,
            masks=masks,
            confidence=0.8,
            model_version="sam2-image-predictor-v1",
            schema_version="sam2",
            label_confidence={k: 0.6 for k in masks},
            debug={"sam2_score": 0.9, "prompt_box": [20, 10, 100, 110], "prompt_points": [[60, 20]]},
        )
        tracked = TrackedHuman(track_id=5, detection=self.det, pose=None, parsed=parsing)
        representation = HumanRepresentationBuilder().build_for_tracked_human(tracked, frame=self.frame)
        self.assertIn("torso", representation.body_parts)
        self.assertEqual(representation.garments, {})

    def test_tracker_propagated_only_keeps_debug_and_copies_masks(self) -> None:
        tracker = SimpleTracker(confidence_decay=0.5)
        source_mask = np.zeros((120, 160), dtype=np.uint8)
        source_mask[10:20, 10:20] = 1
        propagated = ParsedHuman(
            detection_idx=0,
            masks={"person_mask": source_mask},
            confidence=0.8,
            model_version="sam2-image-predictor-v1",
            schema_version="sam2",
            label_confidence={"person_mask": 0.9},
            debug={"prompt_box": [20, 10, 100, 110], "sam2_score": 0.92, "mask_shape": [120, 160]},
        )

        fused = tracker._fuse_parsed(parsed_new=None, parsed_propagated=propagated)
        self.assertIsNotNone(fused)
        assert fused is not None
        self.assertEqual(fused.debug.get("sam2_score"), 0.92)
        self.assertEqual(fused.label_confidence["person_mask"], 0.45)

        propagated.masks["person_mask"][10, 10] = 0
        self.assertEqual(int(fused.masks["person_mask"][10, 10]), 1)

    def test_summary_panel_render(self) -> None:
        scene = SceneFrame(frame_index=0, frame=self.frame, detections=[], poses=[], tracked=[])
        images = OpenCVRenderer().render(scene)
        self.assertIn("summary_panel", images)
        self.assertEqual(images["summary_panel"].ndim, 3)


if __name__ == "__main__":
    unittest.main()
