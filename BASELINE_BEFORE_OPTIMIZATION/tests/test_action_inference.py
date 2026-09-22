"""
Unit tests for Action Inference Engine
"""
import unittest
from src.perception_types import (
    NormalizedFrame, DetectedObject, PoseData, WristPose, HandObjectInteraction
)
from src.temporal_logic import TemporalHistoryBuffer
from src.action_inference import ActionInferenceEngine, ObservedAction


class TestActionInference(unittest.TestCase):

    def setUp(self):
        self.buffer = TemporalHistoryBuffer()
        self.engine = ActionInferenceEngine(buffer=self.buffer)

    def _create_frame(
        self,
        frame_num: int,
        objects=None,
        signals=None,
        right_wrist_xy=None
    ) -> NormalizedFrame:
        if objects is None:
            objects = []
        if signals is None:
            signals = {}

        left_w = WristPose()
        right_w = WristPose(x=right_wrist_xy[0], y=right_wrist_xy[1], confidence=0.9) if right_wrist_xy else WristPose()

        return NormalizedFrame(
            frame=frame_num,
            timestamp=frame_num * 0.033,
            objects=objects,
            pose=PoseData(left_wrist=left_w, right_wrist=right_w),
            hand_object_interaction=HandObjectInteraction(),
            interaction_signals=signals
        )

    def test_spray_plant_inference(self):
        """Test 1: persistent hand near spray + plant -> SPRAY_PLANT evidence."""
        spray = DetectedObject("spray_bottle", 5, 0.9, (100, 100, 150, 150), (125, 125))
        plant = DetectedObject("plant", 4, 0.9, (200, 200, 250, 250), (225, 225))

        signals = {
            "right_hand_near_spray_bottle": True,
            "right_hand_near_plant": True
        }

        action: ObservedAction = None
        for i in range(1, 6):
            frame = self._create_frame(i, objects=[spray, plant], signals=signals)
            action = self.engine.process_frame(frame)

        self.assertEqual(action.action, "SPRAY_PLANT")
        self.assertIn("spray_bottle", action.involved_objects)
        self.assertIn("plant", action.involved_objects)
        self.assertEqual(action.confidence_level, "HIGH")

    def test_pick_spray_lifting_inference(self):
        """Test 2: spray bottle lifted from workplace -> PICK_SPRAY evidence."""
        spray = DetectedObject("spray_bottle", 5, 0.9, (100, 100, 150, 150), (125, 125))
        signals = {"right_hand_near_spray_bottle": True}

        # Wrist moving UP (y decreases)
        for i in range(1, 6):
            y_pos = 500 - (i * 10)
            frame = self._create_frame(i, objects=[spray], signals=signals, right_wrist_xy=(125, y_pos))
            action = self.engine.process_frame(frame)

        self.assertEqual(action.action, "PICK_SPRAY")
        self.assertEqual(action.confidence_level, "HIGH")

    def test_spray_to_workplace_again_inference(self):
        """Test 3: spray bottle returned to workplace -> SPRAY_TO_WORKPLACE_AGAIN evidence."""
        spray = DetectedObject("spray_bottle", 5, 0.9, (100, 100, 150, 150), (125, 125))
        signals = {"right_hand_near_spray_bottle": True}

        # Wrist moving DOWN (y increases)
        for i in range(1, 6):
            y_pos = 400 + (i * 10)
            frame = self._create_frame(i, objects=[spray], signals=signals, right_wrist_xy=(125, y_pos))
            action = self.engine.process_frame(frame)

        self.assertEqual(action.action, "SPRAY_TO_WORKPLACE_AGAIN")

    def test_plant_to_workplace_inference(self):
        """Test 4: plant movement toward workplace."""
        signals = {"right_hand_near_plant": True}
        for i in range(1, 6):
            cx = 400 - (i * 15)  # moving left/down toward workplace
            cy = 300 + (i * 10)
            plant = DetectedObject("plant", 4, 0.9, (cx - 20, cy - 20, cx + 20, cy + 20), (cx, cy))
            frame = self._create_frame(i, objects=[plant], signals=signals)
            action = self.engine.process_frame(frame)

        self.assertEqual(action.action, "PLANT_TO_WORKPLACE")

    def test_plant_to_red_box_inference(self):
        """Test 5: plant movement toward red box."""
        signals = {"right_hand_near_plant": True}
        for i in range(1, 6):
            cx = 200 + (i * 15)  # moving right/up toward red box
            cy = 400 - (i * 10)
            plant = DetectedObject("plant", 4, 0.9, (cx - 20, cy - 20, cx + 20, cy + 20), (cx, cy))
            frame = self._create_frame(i, objects=[plant], signals=signals)
            action = self.engine.process_frame(frame)

        self.assertEqual(action.action, "PLANT_TO_RED_BOX")

    def test_red_box_retrieval_inference(self):
        """Test 6: red box movement out of container."""
        signals = {"right_hand_near_red_box": True}
        for i in range(1, 6):
            cx = 500 - (i * 15)  # moving left/down
            cy = 300 + (i * 10)
            red_box = DetectedObject("red_box", 2, 0.9, (cx - 30, cy - 30, cx + 30, cy + 30), (cx, cy))
            frame = self._create_frame(i, objects=[red_box], signals=signals)
            action = self.engine.process_frame(frame)

        self.assertEqual(action.action, "RETRIEVE_RED_BOX")

    def test_yellow_box_retrieval_inference(self):
        """Test 7: yellow box movement out of container."""
        signals = {"right_hand_near_yellow_box": True}
        for i in range(1, 6):
            cx = 500 - (i * 15)
            cy = 300 + (i * 10)
            yellow_box = DetectedObject("yellow_box", 3, 0.9, (cx - 30, cy - 30, cx + 30, cy + 30), (cx, cy))
            frame = self._create_frame(i, objects=[yellow_box], signals=signals)
            action = self.engine.process_frame(frame)

        self.assertEqual(action.action, "RETRIEVE_YELLOW_BOX")

    def test_temporary_spray_loss_tolerance(self):
        """Test 8: temporary spray_bottle detection loss keeps recent context alive."""
        spray = DetectedObject("spray_bottle", 5, 0.9, (100, 100, 150, 150), (125, 125))
        signals = {"right_hand_near_spray_bottle": True}

        for i in range(1, 4):
            frame = self._create_frame(i, objects=[spray], signals=signals)
            self.engine.process_frame(frame)

        # Brief detection gap (missing spray object)
        gap_frame = self._create_frame(4, objects=[], signals=signals)
        action = self.engine.process_frame(gap_frame)

        # Should still recognize hand handling spray due to grace period / signal
        self.assertNotEqual(action.action, "UNCERTAIN")

    def test_insufficient_evidence_uncertain(self):
        """Test 9: insufficient evidence -> UNCERTAIN."""
        frame = self._create_frame(1, objects=[], signals={})
        action = self.engine.process_frame(frame)

        self.assertEqual(action.action, "UNCERTAIN")
        self.assertEqual(action.confidence_level, "UNCERTAIN")


if __name__ == "__main__":
    unittest.main()
