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


if __name__ == "__main__":
    unittest.main()
