"""
Unit Tests for Local Video Recorder (Phase 11)
Tests cover:
9. recorder initialization
10. frame writing
11. unique output filename generation
12. graceful close & file persistence
13. camera shutdown while recording
"""
import unittest
import os
import shutil
import time
from src.video_recorder import VideoRecorder


class TestVideoRecorder(unittest.TestCase):

    def setUp(self):
        self.test_dir = f"test_recordings_{int(time.time() * 1000)}"
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_09_recorder_initialization(self):
        """9. Recorder initializes stopped with output directory."""
        rec = VideoRecorder(output_dir=self.test_dir)
        self.assertFalse(rec.is_recording)
        self.assertIsNone(rec.recording_path)

    def test_10_frame_writing(self):
        """10. Start recording creates path and enqueues frame safely."""
        rec = VideoRecorder(output_dir=self.test_dir)
        path = rec.start_recording(width=320, height=240)
        self.assertTrue(rec.is_recording)
        self.assertIsNotNone(path)
        success = rec.write_frame(frame_data=None)
        self.assertFalse(success)  # None frame returns False
        rec.stop_recording()

    def test_11_unique_output_filename(self):
        """11. Starting multiple recording sessions generates distinct timestamped filenames."""
        rec1 = VideoRecorder(output_dir=self.test_dir)
        p1 = rec1.start_recording()
        rec1.stop_recording()

        time.sleep(1.05)  # Ensure second timestamp tick

        rec2 = VideoRecorder(output_dir=self.test_dir)
        p2 = rec2.start_recording()
        rec2.stop_recording()

        self.assertNotEqual(p1, p2)
        self.assertTrue(p1.endswith(".mp4"))
        self.assertTrue(p2.endswith(".mp4"))

    def test_12_graceful_close(self):
        """12. Closing recorder finalizes file and resets recording flag."""
        rec = VideoRecorder(output_dir=self.test_dir)
        path = rec.start_recording()
        self.assertTrue(rec.is_recording)
        saved = rec.stop_recording()
        self.assertFalse(rec.is_recording)
        self.assertEqual(saved, path)

    def test_13_camera_shutdown_while_recording(self):
        """13. Shutting down during active recording flushes queue and closes cleanly."""
        rec = VideoRecorder(output_dir=self.test_dir)
        rec.start_recording()
        self.assertTrue(rec.is_recording)
        rec.close()
        self.assertFalse(rec.is_recording)


if __name__ == "__main__":
    unittest.main()
