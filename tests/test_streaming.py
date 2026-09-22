"""
Unit Tests for IP Video Streaming (Phase 12)
Tests cover:
14. stream configuration loading
15. stream initialization
16. stream failure handling
17. stream shutdown
"""
import unittest
import time
from src.streaming import IPStreamer


class TestIPStreaming(unittest.TestCase):

    def test_14_configuration_loading(self):
        """14. IPStreamer loads streaming config correctly."""
        streamer = IPStreamer(config_path="config/streaming.json")
        self.assertIsNotNone(streamer.config)
        self.assertIn("stream_host", streamer.config)
        self.assertIn("stream_port", streamer.config)

    def test_15_stream_initialization(self):
        """15. Stream launches HTTP server on test port."""
        streamer = IPStreamer(config_path="config/streaming.json")
        streamer.config["stream_port"] = 8999  # Isolated test port
        success = streamer.start_stream()
        self.assertTrue(success)
        self.assertTrue(streamer.is_streaming)
        self.assertEqual(streamer.status, "CONNECTED")
        streamer.stop_stream()
        self.assertFalse(streamer.is_streaming)

    def test_16_stream_failure_handling(self):
        """16. Invalid IP address handles stream failure gracefully without crashing."""
        streamer = IPStreamer(config_path="config/streaming.json")
        streamer.config["stream_host"] = "999.999.999.999"  # Invalid host IP to force bind failure
        success = streamer.start_stream()
        self.assertFalse(success)
        self.assertFalse(streamer.is_streaming)
        self.assertEqual(streamer.status, "ERROR")
        self.assertIsNotNone(streamer.error_message)

    def test_17_stream_shutdown(self):
        """17. Stopping stream closes server cleanly."""
        streamer = IPStreamer(config_path="config/streaming.json")
        streamer.config["stream_port"] = 8998
        streamer.start_stream()
        self.assertTrue(streamer.is_streaming)
        streamer.stop_stream()
        self.assertFalse(streamer.is_streaming)
        self.assertEqual(streamer.status, "DISCONNECTED")


if __name__ == "__main__":
    unittest.main()
