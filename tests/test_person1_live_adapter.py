"""
Unit Tests for Person 1 Live Perception Adapter (Phase 10C)
Tests cover:
1. detector model loading (models/best.pt)
2. pose model loading (models/yolo26n-pose.pt)
3. missing model handling
4. corrupted/invalid model handling
5. class-name extraction
6. live detection conversion
7. pose conversion
8. output schema compatibility
9. adapter connected status
10. adapter disconnected status
"""
import unittest
import os
import time
import json
import tempfile
import shutil
from unittest import mock
import numpy as np
from src.person1_live_adapter import Person1LiveAdapter


class MockTensorVal:
    def __init__(self, val):
        self._val = val
    def item(self):
        return self._val
    def tolist(self):
        return self._val
    def __getitem__(self, idx):
        return self

class MockBoxes:
    def __init__(self):
        self.cls = [MockTensorVal(0)]
        self.conf = [MockTensorVal(0.92)]
        self.id = [MockTensorVal(1)]
        self.xyxy = [MockTensorVal([100.0, 100.0, 500.0, 900.0])]

class MockKeypoints:
    def __init__(self):
        kpts = [[0.0, 0.0, 0.0] for _ in range(17)]
        kpts[9] = [400.0, 700.0, 0.85]   # left wrist
        kpts[10] = [450.0, 720.0, 0.88]  # right wrist
        self.data = [kpts]

class MockResult:
    def __init__(self):
        self.boxes = [MockBoxes()]
        self.keypoints = MockKeypoints()

class MockYOLOModel:
    def __init__(self, names=None):
        self.names = names if names else {0: "person", 1: "white_container", 2: "red_box", 3: "yellow_box", 4: "plant", 5: "spray_bottle"}

    def track(self, image, conf=0.35, imgsz=640, persist=True, verbose=False):
        return [MockResult()]

    def __call__(self, image, conf=0.35, imgsz=640, verbose=False):
        return [MockResult()]


class MockWorldBoxes(MockBoxes):
    """Boxes mock carrying a configurable class id."""
    def __init__(self, cls_id):
        super().__init__()
        self.cls = [MockTensorVal(cls_id)]


class MockWorldResult(MockResult):
    """Result mock exposing one box per YOLO-World class id."""
    def __init__(self):
        super().__init__()
        self.boxes = [MockWorldBoxes(i) for i in range(6)]


class MockYOLOWorldModel(MockYOLOModel):
    """Mock YOLO-World model that records set_classes() calls."""
    def __init__(self, names=None):
        super().__init__(names)
        self.set_classes_calls = []

    def set_classes(self, classes):
        self.set_classes_calls.append(list(classes))

    def __call__(self, image, conf=0.35, imgsz=640, verbose=False):
        return [MockWorldResult()]


class TestPerson1LiveAdapter(unittest.TestCase):

    def test_01_detector_model_loading(self):
        """1. Loading real detector model models/best.pt if present."""
        if os.path.exists("models/best.pt"):
            adapter = Person1LiveAdapter(config_path="config/person1_live.json", mock_mode=False)
            self.assertTrue(adapter.is_cv_model_connected)
            self.assertIsNotNone(adapter.model)

    def test_02_pose_model_loading(self):
        """2. Loading real pose model models/yolo26n-pose.pt if present."""
        if os.path.exists("models/yolo26n-pose.pt"):
            adapter = Person1LiveAdapter(config_path="config/person1_live.json", mock_mode=False)
            self.assertIsNotNone(adapter.pose_model)

    def test_03_missing_model_handling(self):
        """3. Missing model file returns PERCEPTION_NOT_CONNECTED without crashing."""
        adapter = Person1LiveAdapter(config_path="non_existent_config.json", mock_mode=False)
        adapter.config["model_path"] = "models/non_existent_yolo.pt"
        adapter._try_load_models()
        self.assertFalse(adapter.is_cv_model_connected)
        self.assertEqual(adapter.adapter_status, "PERCEPTION_NOT_CONNECTED")
        rec = adapter.process_live_frame(frame_num=1, timestamp=time.time())
        self.assertEqual(rec["adapter_status"], "PERCEPTION_NOT_CONNECTED")

    def test_04_corrupted_model_handling(self):
        """4. Corrupted or invalid model file path handled safely."""
        adapter = Person1LiveAdapter(mock_mode=False)
        adapter.config["model_path"] = "config/person1_live.json"  # Invalid model binary
        adapter._try_load_models()
        self.assertFalse(adapter.is_cv_model_connected)
        self.assertEqual(adapter.adapter_status, "PERCEPTION_NOT_CONNECTED")

    def test_05_class_name_extraction(self):
        """5. Class names extracted from model match expected experiment classes."""
        mock_det = MockYOLOModel()
        adapter = Person1LiveAdapter(custom_model=mock_det, mock_mode=False)
        self.assertEqual(adapter.model_classes[0], "person")
        self.assertEqual(adapter.model_classes[1], "white_container")

    def test_06_live_detection_conversion(self):
        """6. Live image matrix conversion produces valid detection bboxes."""
        mock_det = MockYOLOModel()
        adapter = Person1LiveAdapter(custom_model=mock_det, mock_mode=False)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        rec = adapter.process_live_frame(frame_num=10, timestamp=time.time(), image_matrix=img)
        self.assertGreater(len(rec["objects"]), 0)
        self.assertEqual(rec["objects"][0]["class"], "person")
        self.assertEqual(rec["objects"][0]["track_id"], 1)

    def test_07_pose_conversion(self):
        """7. Keypoints converted to left_wrist and right_wrist coordinates."""
        mock_det = MockYOLOModel()
        mock_pose = MockYOLOModel()
        adapter = Person1LiveAdapter(custom_model=mock_det, custom_pose_model=mock_pose, mock_mode=False)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        rec = adapter.process_live_frame(frame_num=10, timestamp=time.time(), image_matrix=img)
        self.assertIn("left_wrist", rec["pose"])
        self.assertIn("right_wrist", rec["pose"])
        self.assertEqual(rec["pose"]["left_wrist"]["x"], 400.0)
        self.assertEqual(rec["pose"]["right_wrist"]["x"], 450.0)

    def test_08_output_schema_compatibility(self):
        """8. Output record strictly contains all mandatory PERSON1_OUTPUT_SPEC keys."""
        adapter = Person1LiveAdapter(mock_mode=True)
        rec = adapter.process_live_frame(frame_num=1, timestamp=1.0)
        required_keys = ["frame", "timestamp", "objects", "pose", "hand_object_interaction", "interaction_signals"]
        for key in required_keys:
            self.assertIn(key, rec)

    def test_09_adapter_connected_status(self):
        """9. Valid model initialization sets adapter_status to CONNECTED."""
        mock_det = MockYOLOModel()
        adapter = Person1LiveAdapter(custom_model=mock_det, mock_mode=False)
        self.assertTrue(adapter.is_cv_model_connected)
        self.assertEqual(adapter.adapter_status, "CONNECTED")

    def test_10_adapter_disconnected_status(self):
        """10. Missing model sets adapter_status to PERCEPTION_NOT_CONNECTED."""
        adapter = Person1LiveAdapter(config_path="non_existent_config.json", mock_mode=False)
        adapter.config["model_path"] = "missing.pt"
        adapter._try_load_models()
        self.assertFalse(adapter.is_cv_model_connected)
        self.assertEqual(adapter.adapter_status, "PERCEPTION_NOT_CONNECTED")

    def test_11_yolo_world_class_configuration(self):
        """11. A model path containing 'world' loads YOLO-World and receives exactly the 6 requested classes."""
        expected_mapping = {
            0: "person",
            1: "white container",
            2: "yellow box",
            3: "red box",
            4: "plant",
            5: "spray bottle"
        }

        temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)
        config_path = os.path.join(temp_dir, "person1_world_smoke.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({
                "model_path": "models/yolov8s-world.pt",
                "pose_model_path": "models/non_existent_pose_smoke.pt",
                "device": "cpu",
                "confidence_threshold": 0.20,
                "image_size": 640,
                "tracking_enabled": False
            }, f)

        mock_world = MockYOLOWorldModel(names=dict(expected_mapping))
        with mock.patch("ultralytics.YOLOWorld", return_value=mock_world):
            adapter = Person1LiveAdapter(config_path=config_path, mock_mode=False)

        self.assertTrue(adapter.is_cv_model_connected)
        self.assertEqual(adapter.adapter_status, "CONNECTED")
        self.assertEqual(mock_world.set_classes_calls, [list(expected_mapping.values())])
        self.assertEqual(len(mock_world.set_classes_calls[0]), 6)
        self.assertEqual(adapter.model_classes, expected_mapping)
        self.assertEqual(adapter.device, "cpu")

    def test_12_yolo_world_class_name_normalization(self):
        """12. YOLO-World class names are normalized to the pipeline naming convention."""
        mock_world = MockYOLOWorldModel(names={
            0: "person", 1: "white container", 2: "yellow box", 3: "red box",
            4: "plant", 5: "spray bottle"
        })
        adapter = Person1LiveAdapter(custom_model=mock_world, mock_mode=False)
        # Red-filled frame so the prototype spray-bottle color check
        # keeps red_box as red_box (deterministic, no real inference).
        img = np.full((480, 640, 3), (0, 0, 255), dtype=np.uint8)
        rec = adapter.process_live_frame(frame_num=1, timestamp=1.0, image_matrix=img)

        detected = {obj["class"] for obj in rec["objects"]}
        self.assertEqual(detected, {
            "person", "white_container", "yellow_box", "red_box", "plant",
            "spray_bottle"
        })


if __name__ == "__main__":
    unittest.main()
