"""
Unit tests for Person1 Input Adapter (Phase 1 verification)
"""
import os
import unittest
from src.input_adapter import Person1Adapter
from src.perception_types import NormalizedFrame


class TestInputAdapter(unittest.TestCase):

    def setUp(self):
        self.sample_json_str = '''{
            "frame": 1,
            "timestamp": 0.034,
            "objects": [
                {"class": "person", "track_id": 1, "confidence": 0.8097, "bbox": [103.12, 373.36, 906.68, 1122.84]},
                {"class": "white_container", "track_id": 2, "confidence": 0.7916, "bbox": [269.06, 898.77, 816.61, 1205.99]}
            ],
            "pose": {
                "person_track_id": 1,
                "left_wrist": {"x": 804.57, "y": 1161.33, "confidence": 0.5087},
                "right_wrist": {"x": 176.62, "y": 1162.65, "confidence": 0.6357}
            },
            "hand_object_interaction": {
                "left": {
                    "white_container": {"near": true, "stable": false, "track_id": 2, "distance_px": 0.0}
                },
                "right": {
                    "white_container": {"near": true, "stable": false, "track_id": 2, "distance_px": 92.44}
                }
            },
            "interaction_signals": {
                "left_hand_near_spray_bottle": false,
                "left_hand_near_plant": false,
                "left_hand_near_red_box": false,
                "left_hand_near_yellow_box": false,
                "left_hand_near_white_container": true,
                "right_hand_near_spray_bottle": false,
                "right_hand_near_plant": false,
                "right_hand_near_red_box": false,
                "right_hand_near_yellow_box": false,
                "right_hand_near_white_container": true
            }
        }'''

    def test_parse_record(self):
        frame: NormalizedFrame = Person1Adapter.parse_record(self.sample_json_str)

        self.assertEqual(frame.frame, 1)
        self.assertAlmostEqual(frame.timestamp, 0.034)
        self.assertEqual(len(frame.objects), 2)
        self.assertEqual(frame.objects[0].class_name, "person")
        self.assertEqual(frame.objects[1].class_name, "white_container")
        self.assertAlmostEqual(frame.pose.left_wrist.x, 804.57)
        self.assertTrue(frame.interaction_signals["left_hand_near_white_container"])
        self.assertTrue(frame.is_hand_near("white_container"))

    def test_stream_dataset_file(self):
        dataset_path = os.path.join("data", "person1_visual_output.jsonl")
        if not os.path.exists(dataset_path):
            self.skipTest(f"Dataset file {dataset_path} not found.")

        count = 0
        for frame in Person1Adapter.stream_file(dataset_path):
            count += 1
            self.assertIsInstance(frame, NormalizedFrame)
            self.assertGreater(frame.frame, 0)
            self.assertGreaterEqual(frame.timestamp, 0.0)

        self.assertEqual(count, 1860, "Should stream all 1860 frames from visual output file.")


if __name__ == "__main__":
    unittest.main()
