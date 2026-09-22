"""
Unit tests for Offline Voice Alert Manager (Part G verification)
"""
import unittest
from src.voice_alert import VoiceAlertManager
from src.state_machine import StateUpdate


class TestVoiceAlert(unittest.TestCase):

    def setUp(self):
        self.vm = VoiceAlertManager(use_mock_tts=True)

    def test_rate_limiting_duplicate_suppression(self):
        """Test 10: Voice alert rate limiting suppresses duplicate warnings."""
        # Speak initial warning
        spk1 = self.vm.speak("Warning. A required step appears to have been skipped.", category="ERROR", timestamp=1.0)
        self.assertTrue(spk1)

        # Immediate repeat at t=1.1s should be suppressed
        spk2 = self.vm.speak("Warning. A required step appears to have been skipped.", category="ERROR", timestamp=1.1)
        self.assertFalse(spk2)

        # After cooldown at t=12.0s, repeat should be allowed
        spk3 = self.vm.speak("Warning. A required step appears to have been skipped.", category="ERROR", timestamp=12.0)
        self.assertTrue(spk3)

    def test_different_categories_generate_new_alerts(self):
        """Test 11: Different error categories generate new alerts without suppression."""
        spk1 = self.vm.speak("Warning. A required step appears to have been skipped.", category="ERROR", timestamp=1.0)
        self.assertTrue(spk1)

        spk2 = self.vm.speak("Warning. Action is out of sequence. Please complete the current step.", category="ERROR", timestamp=1.5)
        self.assertTrue(spk2)

        self.assertEqual(len(self.vm.spoken_history), 2)

    def test_voice_can_be_disabled(self):
        """Test 12: Voice can be disabled."""
        self.vm.voice_enabled = False

        spk = self.vm.speak("Open the white box.", category="STEP", timestamp=1.0)
        self.assertFalse(spk)
        self.assertEqual(len(self.vm.spoken_history), 0)


if __name__ == "__main__":
    unittest.main()
