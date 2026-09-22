"""
Unit Tests for Live System Controls & Failure Isolation (Phase 10B/11/12)
Tests cover:
18. Person 1 unavailable state is visible
19. mock mode must be explicit (no silent fallback)
20. stream failure does not stop experiment processing
21. recording failure does not stop experiment processing
"""
import unittest
import os
import shutil
import time
from src.live_camera import LiveCameraManager
from src.person1_live_adapter import Person1LiveAdapter
from src.video_recorder import VideoRecorder
from src.streaming import IPStreamer
from src.pipeline import ExperimentPipeline


class TestLiveSystemControls(unittest.TestCase):

    def setUp(self):
        self.test_dir = f"test_logs_sys_{int(time.time() * 1000)}"
        os.makedirs(self.test_dir, exist_ok=True)
        self.pipeline = ExperimentPipeline(log_dir=self.test_dir, use_mock_tts=True)

    def tearDown(self):
        if hasattr(self, 'pipeline') and self.pipeline:
            self.pipeline.close()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_18_person1_unavailable_state_visible(self):
        """18. Person 1 perception missing status is reported clearly in adapter status."""
        adapter = Person1LiveAdapter(config_path="non_existent_path.json", mock_mode=False)
        adapter.config["model_path"] = "models/non_existent_model.pt"
        adapter._try_load_models()
        self.assertFalse(adapter.is_cv_model_connected)
        self.assertEqual(adapter.adapter_status, "PERCEPTION_NOT_CONNECTED")

    def test_19_mock_mode_must_be_explicit(self):
        """19. LIVE_CAMERA mode with non-existent camera sets CAMERA_UNAVAILABLE and does NOT silently switch to MOCK."""
        cam_mgr = LiveCameraManager(
            device_index=99999,  # Non-existent physical webcam
            pipeline=self.pipeline,
            adapter=Person1LiveAdapter(mock_mode=False),
            mock_camera=False  # User selected LIVE_CAMERA
        )
        started = cam_mgr.start_camera()
        self.assertFalse(started)
        self.assertEqual(cam_mgr.camera_status, "CAMERA_UNAVAILABLE")
        self.assertFalse(cam_mgr.is_running)

    def test_20_stream_failure_does_not_stop_experiment_processing(self):
        """20. Failed IP stream server does not halt downstream pipeline processing."""
        bad_streamer = IPStreamer()
        bad_streamer.config["stream_host"] = "999.999.999.999"  # Force bind error
        bad_streamer.start_stream()
        self.assertEqual(bad_streamer.status, "ERROR")

        adapter = Person1LiveAdapter(mock_mode=True)
        cam_mgr = LiveCameraManager(
            pipeline=self.pipeline,
            adapter=adapter,
            streamer=bad_streamer,
            mock_camera=True
        )

        updates = []
        cam_mgr.on_frame_callback = lambda img, u, f, a, m: updates.append(u)

        cam_mgr.start_camera()
        cam_mgr.start_experiment()
        time.sleep(0.15)
        cam_mgr.stop_camera()

        self.assertGreater(len(updates), 0)
        self.assertEqual(cam_mgr.pipeline.state_machine.status, "ACTIVE")

    def test_21_recording_failure_does_not_stop_experiment_processing(self):
        """21. Faulty video recorder does not break downstream pipeline processing."""
        bad_recorder = VideoRecorder(output_dir="invalid_dir_#$%\x00")
        adapter = Person1LiveAdapter(mock_mode=True)
        cam_mgr = LiveCameraManager(
            pipeline=self.pipeline,
            adapter=adapter,
            recorder=bad_recorder,
            mock_camera=True
        )

        updates = []
        cam_mgr.on_frame_callback = lambda img, u, f, a, m: updates.append(u)

        cam_mgr.start_camera()
        cam_mgr.start_experiment()
        time.sleep(0.15)
        cam_mgr.stop_camera()

        self.assertGreater(len(updates), 0)


if __name__ == "__main__":
    unittest.main()
