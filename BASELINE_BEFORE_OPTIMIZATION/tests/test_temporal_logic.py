"""
Unit tests for Temporal History Buffer
"""
import unittest
from src.perception_types import (
    NormalizedFrame, DetectedObject, PoseData, WristPose, HandObjectInteraction, InteractionDetail
)
from src.temporal_logic import TemporalHistoryBuffer


class TestTemporalLogic(unittest.TestCase):

    def setUp(self):
        self.buffer = TemporalHistoryBuffer()

    def _create_mock_frame(
        self,
        frame_id: int,
        timestamp: float,
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
            frame=frame_id,
            timestamp=timestamp,
            objects=objects,
            pose=PoseData(left_wrist=left_w, right_wrist=right_w),
            hand_object_interaction=HandObjectInteraction(),
            interaction_signals=signals
        )

    def test_temporary_detection_loss_grace_period(self):
        """Tests that temporary CV loss of an object is tolerated during grace period."""
        spray_obj = DetectedObject("spray_bottle", 10, 0.9, (100, 100, 150, 150), (125, 125))

        # Add 3 frames with spray bottle
        for i in range(1, 4):
            self.buffer.add_frame(self._create_mock_frame(i, i * 0.033, objects=[spray_obj]))

        self.assertTrue(self.buffer.is_object_recently_present("spray_bottle", grace_period_frames=5))

        # Add 3 frames WITHOUT spray bottle (detection gap)
        for i in range(4, 7):
            self.buffer.add_frame(self._create_mock_frame(i, i * 0.033, objects=[]))

        # Should still be True because grace period is 5
        self.assertTrue(self.buffer.is_object_recently_present("spray_bottle", grace_period_frames=5))

        # Add 5 more frames WITHOUT spray bottle (exceeding grace period)
        for i in range(7, 12):
            self.buffer.add_frame(self._create_mock_frame(i, i * 0.033, objects=[]))

        self.assertFalse(self.buffer.is_object_recently_present("spray_bottle", grace_period_frames=5))

    def test_object_displacement_calculation(self):
        """Tests displacement and directional movement calculation over sliding window."""
        for i in range(1, 6):
            # Plant moving down and left (center x decreases, y increases)
            cx = 500 - (i * 20)
            cy = 300 + (i * 15)
            plant_obj = DetectedObject("plant", 4, 0.85, (cx - 20, cy - 20, cx + 20, cy + 20), (cx, cy))
            self.buffer.add_frame(self._create_mock_frame(i, i * 0.033, objects=[plant_obj]))

        disp = self.buffer.get_object_displacement("plant")
        self.assertIsNotNone(disp)
        self.assertTrue(disp["is_moving_left"])
        self.assertTrue(disp["is_moving_down"])
        self.assertGreater(disp["distance_px"], 50.0)


if __name__ == "__main__":
    unittest.main()
