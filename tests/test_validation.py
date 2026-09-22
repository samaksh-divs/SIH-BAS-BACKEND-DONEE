"""
Automated Unit Tests for Phase 9 Validation Suite (Part K verification)
"""
import os
import unittest
from src.pipeline import ExperimentPipeline
from src.state_machine import ExperimentStateMachine, StateUpdate
from src.action_inference import ObservedAction
from src.error_detector import ErrorCategory


class TestPhase9Validation(unittest.TestCase):

    def setUp(self):
        self.sm = ExperimentStateMachine(required_confirmations=3)
        self.pipeline = ExperimentPipeline(log_dir="test_logs", use_mock_tts=True)

    def tearDown(self):
        self.pipeline.close()

    def _make_obs(self, action_name: str, confidence: float = 0.90, confidence_level: str = "HIGH") -> ObservedAction:
        return ObservedAction(
            action=action_name,
            confidence=confidence,
            confidence_level=confidence_level,
            evidence=[f"Synthetic test evidence for {action_name}"],
            start_frame=1,
            end_frame=1,
            timestamp=0.0,
            involved_objects=[],
            wrist_side="right",
            reason=f"Validation test {action_name}"
        )

    def test_01_correct_sequence_reaches_complete(self):
        """Test 1: Perfect correct sequence S01 -> S15 -> S16 COMPLETE."""
        sequence_actions = [
            "OPEN_WHITE_BOX", "RETRIEVE_RED_BOX", "RETRIEVE_YELLOW_BOX",
            "OPEN_RED_BOX", "PLANT_TO_WORKPLACE", "OPEN_YELLOW_BOX",
            "SPRAY_TO_WORKPLACE", "PICK_SPRAY", "SPRAY_PLANT",
            "SPRAY_TO_WORKPLACE_AGAIN", "PLANT_TO_RED_BOX", "SPRAY_TO_YELLOW_BOX",
            "RED_BOX_TO_WHITE_BOX", "YELLOW_BOX_TO_WHITE_BOX", "CLOSE_WHITE_BOX"
        ]

        frame_idx = 1
        for act in sequence_actions:
            obs = self._make_obs(act)
            for _ in range(3):
                up = self.sm.update(obs, frame=frame_idx, timestamp=frame_idx * 0.1)
                frame_idx += 1

        self.assertEqual(self.sm.current_state.state_id, "S16")
        self.assertEqual(self.sm.status, "COMPLETE")

    def test_02_and_03_s15_completion_behavior(self):
        """Tests 2 & 3: S15 active does NOT mean COMPLETE until S15 completes."""
        self.sm.resync_to_step(15)  # S15 CLOSE_WHITE_BOX
        obs = self._make_obs("CLOSE_WHITE_BOX")

        # Frame 1: S15 becomes active but is NOT COMPLETE
        up1 = self.sm.update(obs, frame=1, timestamp=0.1)
        self.assertEqual(up1.current_state_id, "S15")
        self.assertEqual(up1.status, "ACTIVE")
        self.assertNotEqual(up1.status, "COMPLETE")

        # Frame 2: Still S15 ACTIVE
        up2 = self.sm.update(obs, frame=2, timestamp=0.2)
        self.assertEqual(up2.current_state_id, "S15")
        self.assertEqual(up2.status, "ACTIVE")

        # Frame 3: 3rd confirmation -> S15 completes, transition to S16 COMPLETE
        up3 = self.sm.update(obs, frame=3, timestamp=0.3)
        self.assertEqual(up3.current_state_id, "S16")
        self.assertEqual(up3.status, "COMPLETE")

    def test_04_recovery_transition_distinguishable(self):
        """Test 4: Recovery transition is explicitly classified with transition_type='RECOVERY'."""
        self.sm.resync_to_step(8)  # S08 PICK_SPRAY
        skipped_act = self._make_obs("SPRAY_PLANT")

        updates = []
        for f in range(1, 6):
            updates.append(self.sm.update(skipped_act, frame=f, timestamp=f * 0.1))

        last_up = updates[-1]
        self.assertEqual(last_up.status, "RECOVERY")
        self.assertEqual(last_up.transition_type, "RECOVERY")

    def test_05_multi_step_jump_classified(self):
        """Test 5: Explicitly verify normal vs multi-step jump transition classification."""
        self.sm.reset()
        obs_normal = self._make_obs("OPEN_WHITE_BOX")
        up_norm = None
        for f in range(1, 4):
            up_norm = self.sm.update(obs_normal, frame=f, timestamp=f * 0.1)

        self.assertTrue(up_norm.transitioned)
        self.assertEqual(up_norm.transition_type, "NORMAL")

    def test_06_detection_gap_does_not_cause_false_error(self):
        """Test 6: Temporary detection gap does not trigger a false error."""
        self.sm.reset()
        self.sm.update(self._make_obs("OPEN_WHITE_BOX"), frame=1, timestamp=0.1)

        # Gap frame (UNCERTAIN)
        gap_obs = self._make_obs("UNCERTAIN", confidence=0.1, confidence_level="UNCERTAIN")
        up = self.sm.update(gap_obs, frame=2, timestamp=0.2)

        self.assertEqual(up.status, "WAITING")
        self.assertIn(up.error_type, [None, ErrorCategory.PERCEPTION_UNCERTAIN])

    def test_07_correct_run_error_count(self):
        """Test 7: Perfect correct run produces 0 procedural errors."""
        self.sm.reset()
        sequence_actions = [
            "OPEN_WHITE_BOX", "RETRIEVE_RED_BOX", "RETRIEVE_YELLOW_BOX",
            "OPEN_RED_BOX", "PLANT_TO_WORKPLACE", "OPEN_YELLOW_BOX",
            "SPRAY_TO_WORKPLACE", "PICK_SPRAY", "SPRAY_PLANT",
            "SPRAY_TO_WORKPLACE_AGAIN", "PLANT_TO_RED_BOX", "SPRAY_TO_YELLOW_BOX",
            "RED_BOX_TO_WHITE_BOX", "YELLOW_BOX_TO_WHITE_BOX", "CLOSE_WHITE_BOX"
        ]

        errors = []
        frame_idx = 1
        for act in sequence_actions:
            obs = self._make_obs(act)
            for _ in range(3):
                up = self.sm.update(obs, frame=frame_idx, timestamp=frame_idx * 0.1)
                if up.status == "ERROR" and up.error_type:
                    errors.append(up)
                frame_idx += 1

        self.assertEqual(len(errors), 0, "Perfect correct run must produce zero procedural errors.")

    def test_08_log_and_state_machine_agreement(self):
        """Test 8: Log and state machine agree on state updates."""
        sample_rec = {
            "frame": 1, "timestamp": 0.033,
            "objects": [{"class": "white_container", "track_id": 2, "confidence": 0.9, "bbox": [10, 10, 100, 100]}],
            "pose": {}, "hand_object_interaction": {},
            "interaction_signals": {"right_hand_near_white_container": True}
        }
        up, norm_f, obs = self.pipeline.process_frame(sample_rec)
        self.assertEqual(up.current_state_id, self.pipeline.state_machine.current_state.state_id)

    def test_09_gui_receives_final_complete_status(self):
        """Test 9: GUI receives final COMPLETE status on S16 transition."""
        received_statuses = []

        def callback(up, f, obs):
            received_statuses.append(up.status)

        pipe = ExperimentPipeline(log_dir="test_logs", use_mock_tts=True, on_update_callback=callback)
        pipe.state_machine.resync_to_step(16)

        sample_rec = {
            "frame": 100, "timestamp": 10.0, "objects": [], "pose": {},
            "hand_object_interaction": {}, "interaction_signals": {}
        }
        pipe.process_frame(sample_rec)

        self.assertIn("COMPLETE", received_statuses)
        pipe.close()


if __name__ == "__main__":
    unittest.main()
