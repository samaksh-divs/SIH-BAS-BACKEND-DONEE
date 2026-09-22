"""
Unit tests for Experiment State Machine & Procedural Error Detection (Phase 3 & 4)
"""
import unittest
from src.action_inference import ObservedAction
from src.state_machine import ExperimentStateMachine, StateUpdate
from src.error_detector import ErrorCategory


class TestStateMachine(unittest.TestCase):

    def setUp(self):
        self.sm = ExperimentStateMachine(
            sequence_config_path="config/experiment_sequence.json",
            required_confirmations=3,
            error_cooldown_frames=10
        )

    def _make_action(self, action_name: str, confidence: float = 0.90, confidence_level: str = "HIGH") -> ObservedAction:
        return ObservedAction(
            action=action_name,
            confidence=confidence,
            confidence_level=confidence_level,
            evidence=[f"Mock evidence for {action_name}"],
            start_frame=1,
            end_frame=1,
            timestamp=0.0,
            involved_objects=[],
            wrist_side="right",
            reason=f"Synthetic test for {action_name}"
        )

    def test_01_single_step_transition(self):
        """TEST 1: Correct S01 -> S02 transition after required confirmations."""
        action = self._make_action("OPEN_WHITE_BOX")

        # Frame 1
        up1 = self.sm.update(action, frame=1, timestamp=0.033)
        self.assertEqual(up1.current_state_id, "S01")
        self.assertFalse(up1.transitioned)

        # Frame 2
        up2 = self.sm.update(action, frame=2, timestamp=0.066)
        self.assertEqual(up2.current_state_id, "S01")

        # Frame 3 (3rd confirmation -> transition to S02)
        up3 = self.sm.update(action, frame=3, timestamp=0.099)
        self.assertTrue(up3.transitioned)
        self.assertEqual(up3.previous_state, "S01")
        self.assertEqual(up3.current_state_id, "S02")
        self.assertEqual(up3.expected_action, "RETRIEVE_RED_BOX")

    def test_02_full_sequence(self):
        """TEST 2: Complete sequence S01 -> S15 -> COMPLETE."""
        sequence_actions = [
            "OPEN_WHITE_BOX", "RETRIEVE_RED_BOX", "RETRIEVE_YELLOW_BOX",
            "OPEN_RED_BOX", "PLANT_TO_WORKPLACE", "OPEN_YELLOW_BOX",
            "SPRAY_TO_WORKPLACE", "PICK_SPRAY", "SPRAY_PLANT",
            "SPRAY_TO_WORKPLACE_AGAIN", "PLANT_TO_RED_BOX", "SPRAY_TO_YELLOW_BOX",
            "RED_BOX_TO_WHITE_BOX", "YELLOW_BOX_TO_WHITE_BOX", "CLOSE_WHITE_BOX"
        ]

        frame_idx = 1
        for act in sequence_actions:
            obs = self._make_action(act)
            for _ in range(3):
                up = self.sm.update(obs, frame=frame_idx, timestamp=frame_idx * 0.1)
                frame_idx += 1

        self.assertEqual(self.sm.current_state.state_id, "S16")
        self.assertEqual(self.sm.status, "COMPLETE")

    def test_03_out_of_sequence_error(self):
        """TEST 3: S02 expected but S03 observed -> OUT_OF_SEQUENCE."""
        # Advance to S02
        for i in range(1, 4):
            self.sm.update(self._make_action("OPEN_WHITE_BOX"), frame=i, timestamp=i * 0.1)

        self.assertEqual(self.sm.current_state.state_id, "S02")

        # Now observe RETRIEVE_YELLOW_BOX (S03) instead of RETRIEVE_RED_BOX (S02)
        up = self.sm.update(self._make_action("RETRIEVE_YELLOW_BOX"), frame=4, timestamp=0.4)
        self.assertEqual(up.status, "ERROR")
        self.assertEqual(up.error_type, ErrorCategory.SKIPPED_STEP)

    def test_04_skipped_step_error(self):
        """TEST 4: S08 expected but S09 observed -> SKIPPED_STEP."""
        self.sm.resync_to_step(8)  # S08 PICK_SPRAY
        self.assertEqual(self.sm.current_state.state_id, "S08")

        up = self.sm.update(self._make_action("SPRAY_PLANT"), frame=10, timestamp=1.0)
        self.assertEqual(up.status, "ERROR")
        self.assertEqual(up.error_type, ErrorCategory.SKIPPED_STEP)

    def test_05_temporary_detection_gap(self):
        """TEST 5: Correct action with temporary detection gap stays in current state without false error."""
        self.sm.update(self._make_action("OPEN_WHITE_BOX"), frame=1, timestamp=0.1)

        # Gap frame (UNCERTAIN)
        gap_action = self._make_action("UNCERTAIN", confidence=0.1, confidence_level="UNCERTAIN")
        up = self.sm.update(gap_action, frame=2, timestamp=0.2)

        self.assertEqual(up.current_state_id, "S01")
        self.assertEqual(up.status, "WAITING")
        self.assertIn(up.error_type, [None, ErrorCategory.PERCEPTION_UNCERTAIN])

    def test_06_low_confidence_uncertain(self):
        """TEST 6: Low-confidence observation -> PERCEPTION_UNCERTAIN."""
        low_conf = self._make_action("OPEN_WHITE_BOX", confidence=0.2, confidence_level="UNCERTAIN")
        up = self.sm.update(low_conf, frame=1, timestamp=0.1)

        self.assertEqual(up.status, "WAITING")
        self.assertEqual(up.error_type, ErrorCategory.PERCEPTION_UNCERTAIN)

    def test_07_no_progress_timeout(self):
        """TEST 7: No progress beyond timeout -> INACTION / WAITING."""
        # Initialize entry timestamp at t=0.0
        self.sm.update(self._make_action("UNCERTAIN", confidence=0.1, confidence_level="UNCERTAIN"), frame=1, timestamp=0.0)

        # Send frame at t=35.0 (exceeding 30.0s timeout)
        up = self.sm.update(self._make_action("UNCERTAIN", confidence=0.1, confidence_level="UNCERTAIN"), frame=100, timestamp=35.0)

        self.assertEqual(up.status, "WAITING")
        self.assertEqual(up.error_type, ErrorCategory.INACTION)

    def test_08_debounced_error_events(self):
        """TEST 8: Same bad observation repeated over many frames -> debounced error messages."""
        self.sm.resync_to_step(2)  # S02 RETRIEVE_RED_BOX
        bad = self._make_action("SPRAY_PLANT")

        # Frame 1: initial error
        up1 = self.sm.update(bad, frame=1, timestamp=0.1)
        self.assertEqual(up1.status, "ERROR")
        self.assertIn("Warning", up1.message)

        # Frame 2: debounced error (message reverts to standard step prompt)
        up2 = self.sm.update(bad, frame=2, timestamp=0.2)
        self.assertEqual(up2.status, "ERROR")
        self.assertEqual(up2.message, self.sm.current_state.message)

    def test_09_recovery_resynchronization(self):
        """TEST 9: Persistent out-of-order action triggers RECOVERY resynchronization."""
        self.sm.resync_to_step(8)  # S08 PICK_SPRAY
        skipped_act = self._make_action("SPRAY_PLANT")

        # Send SPRAY_PLANT for 5 consecutive frames (persisting despite error alert)
        updates = []
        for f in range(1, 6):
            updates.append(self.sm.update(skipped_act, frame=f, timestamp=f * 0.1))

        # 5th frame triggers resynchronization to S09
        last_up = updates[-1]
        self.assertEqual(last_up.status, "RECOVERY")
        self.assertTrue(last_up.transitioned)
        self.assertEqual(self.sm.current_state.state_id, "S09")

    def test_10_pause_and_resume(self):
        """TEST 10: Pause and resume control."""
        self.sm.pause()
        up_p = self.sm.update(self._make_action("OPEN_WHITE_BOX"), frame=1, timestamp=0.1)
        self.assertEqual(up_p.status, "PAUSED")

        self.sm.resume()
        up_r = self.sm.update(self._make_action("OPEN_WHITE_BOX"), frame=2, timestamp=0.2)
        self.assertEqual(up_r.status, "ACTIVE")

    def test_11_reset(self):
        """TEST 11: Reset state machine."""
        self.sm.resync_to_step(10)
        self.assertEqual(self.sm.current_state.state_id, "S10")

        self.sm.reset()
        self.assertEqual(self.sm.current_state.state_id, "S01")
        self.assertEqual(self.sm.status, "START")

    def test_12_complete_is_terminal(self):
        """TEST 12: COMPLETE state is terminal."""
        self.sm.resync_to_step(16)  # S16 COMPLETE
        up = self.sm.update(self._make_action("OPEN_WHITE_BOX"), frame=100, timestamp=10.0)

        self.assertEqual(up.status, "COMPLETE")
        self.assertFalse(up.transitioned)


if __name__ == "__main__":
    unittest.main()
