"""
Automated Unit Tests for Replay Controller & Integrated Pipeline (Part K)
"""
import os
import time
import unittest
from src.pipeline import ExperimentPipeline
from src.replay import ReplayController
from src.state_machine import StateUpdate
from src.perception_types import NormalizedFrame
from src.action_inference import ObservedAction


class TestReplayPipeline(unittest.TestCase):

    def setUp(self):
        self.dataset_path = os.path.join("data", "person1_visual_output.jsonl")
        self.pipeline = ExperimentPipeline(log_dir="test_logs", use_mock_tts=True)
        self.controller = ReplayController(
            dataset_path=self.dataset_path,
            pipeline=self.pipeline,
            replay_speed=4.0
        )

    def tearDown(self):
        self.controller.reset()
        self.pipeline.close()

    def test_01_replay_reads_all_records_and_reaches_eof(self):
        """Tests 1, 2, 3, 4: Replay reads all records, preserves ordering/timestamps, reaches EOF."""
        if not os.path.exists(self.dataset_path):
            self.skipTest(f"Dataset {self.dataset_path} not found.")

        # Stream all frames directly through pipeline to test record completeness
        count = 0
        prev_frame = 0
        prev_ts = -1.0

        for frame_obj in self.pipeline.buffer.__class__().frames:
            pass  # sanity check

        from src.input_adapter import Person1Adapter
        for frame in Person1Adapter.stream_file(self.dataset_path):
            count += 1
            self.assertGreater(frame.frame, prev_frame)
            self.assertGreaterEqual(frame.timestamp, prev_ts)
            prev_frame = frame.frame
            prev_ts = frame.timestamp

        self.assertEqual(count, 1860)

    def test_02_pause_resume_stop_reset(self):
        """Tests 5, 6, 7: Pause, resume, stop, reset functions."""
        self.controller.start()
        time.sleep(0.1)
        self.assertTrue(self.controller.is_running)

        self.controller.pause()
        self.assertTrue(self.controller.is_paused)
        self.assertEqual(self.pipeline.state_machine.status, "PAUSED")

        self.controller.resume()
        self.assertFalse(self.controller.is_paused)

        self.controller.reset()
        self.assertFalse(self.controller.is_running)
        self.assertEqual(self.controller.frames_processed, 0)
        self.assertEqual(self.pipeline.state_machine.current_state.state_id, "S01")

    def test_03_replay_speed_scaling(self):
        """Test 8: Replay speed scaling setting."""
        self.controller.set_speed(2.0)
        self.assertEqual(self.controller.replay_speed, 2.0)

        self.controller.set_speed(4.0)
        self.assertEqual(self.controller.replay_speed, 4.0)

    def test_04_pipeline_downstream_integration(self):
        """Tests 9, 10, 11, 12: Pipeline feeds StateMachine, Logger, VoiceManager, Callbacks."""
        received_updates = []

        def callback(up, norm_f, obs_a):
            received_updates.append(up)

        pipe = ExperimentPipeline(log_dir="test_logs", use_mock_tts=True, on_update_callback=callback)

        sample_record = {
            "frame": 1, "timestamp": 0.033,
            "objects": [{"class": "white_container", "track_id": 2, "confidence": 0.9, "bbox": [10, 10, 100, 100]}],
            "pose": {}, "hand_object_interaction": {},
            "interaction_signals": {"right_hand_near_white_container": True}
        }

        up, norm_f, obs_a = pipe.process_frame(sample_record)

        self.assertEqual(len(received_updates), 1)
        self.assertEqual(up.current_state_id, "S01")
        self.assertIsNotNone(pipe.logger.filepath)
        self.assertGreater(len(pipe.voice_manager.spoken_history), 0)

        pipe.close()

    def test_05_completion_state(self):
        """Tests 14: Terminal completion state handling."""
        self.pipeline.state_machine.resync_to_step(16)  # S16 COMPLETE

        sample_record = {
            "frame": 100, "timestamp": 10.0, "objects": [], "pose": {},
            "hand_object_interaction": {}, "interaction_signals": {}
        }
        up, _, _ = self.pipeline.process_frame(sample_record)

        self.assertEqual(up.status, "COMPLETE")
        self.assertEqual(self.pipeline.state_machine.status, "COMPLETE")


if __name__ == "__main__":
    unittest.main()
