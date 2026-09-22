"""
Unit Tests for Phase 10 — Live Camera Integration & Controls
Tests cover:
1. camera initialization failure
2. camera start
3. camera stop
4. frame numbering
5. live timestamps
6. camera disconnect / mock fallback
7. pipeline receives live records
8. pause
9. resume
10. reset
11. experiment start
12. experiment completion
13. graceful shutdown
"""
import unittest
import time
import os
import shutil
from src.live_camera import LiveCameraManager
from src.person1_live_adapter import Person1LiveAdapter
from src.pipeline import ExperimentPipeline
from src.state_machine import StateUpdate
from src.perception_types import NormalizedFrame
from src.action_inference import ObservedAction


class TestLiveCameraIntegration(unittest.TestCase):

    def setUp(self):
        self.test_log_dir = f"test_logs_live_camera_{int(time.time() * 1000)}"
        os.makedirs(self.test_log_dir, exist_ok=True)

        self.pipeline = ExperimentPipeline(log_dir=self.test_log_dir, use_mock_tts=True)
        self.adapter = Person1LiveAdapter(mock_mode=True)
        self.camera_mgr = LiveCameraManager(
            device_index=999,  # Non-existent physical device index to force mock fallback / test safety
            pipeline=self.pipeline,
            adapter=self.adapter,
            mock_camera=True
        )

    def tearDown(self):
        if self.camera_mgr.is_running:
            self.camera_mgr.stop_camera()
        if hasattr(self, 'pipeline') and self.pipeline:
            self.pipeline.close()
        time.sleep(0.05)
        if hasattr(self, 'test_log_dir') and os.path.exists(self.test_log_dir):
            shutil.rmtree(self.test_log_dir, ignore_errors=True)

    def test_01_camera_initialization_failure_fallback(self):
        """1. Camera initialization failure (e.g. invalid device index sets status to CAMERA_UNAVAILABLE and stays stopped)."""
        mgr = LiveCameraManager(device_index=9999, mock_camera=False)
        started = mgr.start_camera()
        self.assertFalse(started)
        self.assertEqual(mgr.camera_status, "CAMERA_UNAVAILABLE")
        self.assertFalse(mgr.is_running)
        mgr.stop_camera()

    def test_02_camera_start(self):
        """2. Camera start sets flags and launches thread."""
        self.assertFalse(self.camera_mgr.is_running)
        success = self.camera_mgr.start_camera()
        self.assertTrue(success)
        self.assertTrue(self.camera_mgr.is_running)
        self.assertFalse(self.camera_mgr.is_stopped)

    def test_03_camera_stop(self):
        """3. Camera stop safely stops capture loop."""
        self.camera_mgr.start_camera()
        self.assertTrue(self.camera_mgr.is_running)
        self.camera_mgr.stop_camera()
        self.assertFalse(self.camera_mgr.is_running)
        self.assertTrue(self.camera_mgr.is_stopped)

    def test_04_frame_numbering(self):
        """4. Frame numbering increments monotonically."""
        received_frames = []

        def frame_cb(img, update, norm_frame, obs_action, metrics):
            received_frames.append(norm_frame.frame)

        self.camera_mgr.on_frame_callback = frame_cb
        self.camera_mgr.start_camera()
        time.sleep(0.15)
        self.camera_mgr.stop_camera()

        self.assertGreater(len(received_frames), 0)
        self.assertEqual(received_frames, sorted(received_frames))
        self.assertEqual(received_frames[0], 1)
        if len(received_frames) > 1:
            self.assertEqual(received_frames[1], 2)

    def test_05_live_timestamps(self):
        """5. Live timestamps are real-time, non-zero, and increasing."""
        timestamps = []

        def frame_cb(img, update, norm_frame, obs_action, metrics):
            timestamps.append(norm_frame.timestamp)

        self.camera_mgr.on_frame_callback = frame_cb
        self.camera_mgr.start_camera()
        time.sleep(0.15)
        self.camera_mgr.stop_camera()

        self.assertGreater(len(timestamps), 0)
        for ts in timestamps:
            self.assertGreater(ts, 1000000000.0)  # Standard epoch timestamp in seconds
        if len(timestamps) > 1:
            self.assertGreaterEqual(timestamps[1], timestamps[0])

    def test_06_camera_disconnect_handling(self):
        """6. Handling perception unconnection / camera frame loss without pipeline crash."""
        disconnected_adapter = Person1LiveAdapter(mock_mode=False)  # Model not loaded
        rec = disconnected_adapter.process_live_frame(frame_num=10, timestamp=time.time(), image_matrix=None)
        self.assertEqual(rec["adapter_status"], "PERCEPTION_NOT_CONNECTED")
        # Pipeline processing should handle empty perception records gracefully
        update, norm_frame, obs_action = self.pipeline.process_frame(rec)
        self.assertIsNotNone(update)

    def test_07_pipeline_receives_live_records(self):
        """7. Downstream pipeline receives live perception records from camera loop."""
        updates = []

        def frame_cb(img, update, norm_frame, obs_action, metrics):
            updates.append(update)

        self.camera_mgr.on_frame_callback = frame_cb
        self.camera_mgr.start_camera()
        time.sleep(0.15)
        self.camera_mgr.stop_camera()

        self.assertGreater(len(updates), 0)
        self.assertIsInstance(updates[0], StateUpdate)

    def test_08_pause(self):
        """8. Pause experiment halts state machine progression."""
        self.camera_mgr.start_experiment()
        self.camera_mgr.pause_experiment()
        self.assertTrue(self.camera_mgr.is_paused)
        self.assertEqual(self.pipeline.state_machine.status, "PAUSED")

    def test_09_resume(self):
        """9. Resume experiment continues state machine execution."""
        self.camera_mgr.start_experiment()
        self.camera_mgr.pause_experiment()
        self.camera_mgr.resume_experiment()
        self.assertFalse(self.camera_mgr.is_paused)
        self.assertEqual(self.pipeline.state_machine.status, "ACTIVE")

    def test_10_reset(self):
        """10. Reset clears state machine, temporal history, metrics, and logs."""
        self.camera_mgr.start_experiment()
        self.camera_mgr.reset_experiment()
        self.assertFalse(self.camera_mgr.is_experiment_started)
        self.assertEqual(self.camera_mgr.frames_captured, 0)
        self.assertEqual(self.pipeline.state_machine.current_state.state_id, "S01")

    def test_11_experiment_start(self):
        """11. START EXPERIMENT initializes state machine and logs EXPERIMENT_STARTED event."""
        self.camera_mgr.start_experiment()
        self.assertTrue(self.camera_mgr.is_experiment_started)
        self.assertEqual(self.pipeline.state_machine.status, "ACTIVE")
        self.assertEqual(self.pipeline.state_machine.current_state.state_id, "S01")

    def test_12_experiment_completion(self):
        """12. Experiment completes cleanly when state reaches S16 COMPLETE."""
        self.pipeline.state_machine.current_index = 14  # S15 index
        obs = ObservedAction(
            action="CLOSE_WHITE_BOX",
            confidence=0.95,
            confidence_level="HIGH",
            evidence=["Manual test trigger"],
            start_frame=100,
            end_frame=100,
            timestamp=time.time(),
            involved_objects=["white_container"],
            wrist_side="right",
            reason="Test completion trigger"
        )
        # Run 3 confirmation frames to trigger state transition to S16
        for i in range(3):
            update = self.pipeline.state_machine.update(observed=obs, frame=100 + i, timestamp=time.time() + i)
        self.assertEqual(update.current_state_id, "S16")
        self.assertEqual(update.status, "COMPLETE")

    def test_13_graceful_shutdown(self):
        """13. System shuts down camera thread cleanly without deadlocks."""
        self.camera_mgr.start_camera()
        self.assertTrue(self.camera_mgr.is_running)
        self.camera_mgr.stop_camera()
        self.assertFalse(self.camera_mgr.is_running)
        self.assertTrue(self.camera_mgr.is_stopped)


if __name__ == "__main__":
    unittest.main()
