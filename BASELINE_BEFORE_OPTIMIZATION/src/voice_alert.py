"""
Offline Voice Alert Manager
Provides rate-limited offline text-to-speech alerts using pyttsx3 (with fallback for mock/test environments).
"""
import json
import os
import threading
import time
from typing import Dict, List, Optional, Tuple, Any

from src.state_machine import StateUpdate

try:
    import pyttsx3
    HAS_PYTTSX3 = True
except ImportError:
    HAS_PYTTSX3 = False


class VoiceAlertManager:
    """
    Manages rate-limited offline voice prompts for step instructions and warning alerts.
    """

    DEFAULT_ERROR_MESSAGES = {
        "SKIPPED_STEP": "Warning. A required step appears to have been skipped.",
        "OUT_OF_SEQUENCE": "Warning. Action is out of sequence. Please complete the current step.",
        "WRONG_OBJECT": "Warning. Incorrect object detected.",
        "WRONG_TARGET": "Warning. Incorrect target detected.",
        "INACTION": "Please continue with the current step.",
        "PERCEPTION_UNCERTAIN": "Perception is uncertain. Please continue carefully.",
        "RECOVERY": "The system has resynchronized to the detected step.",
        "COMPLETE": "Experiment complete."
    }

    def __init__(self, config_path: Optional[str] = "config/thresholds.json", use_mock_tts: bool = False):
        self.voice_enabled = True
        self.voice_cooldown_seconds = 5.0
        self.repeat_error_after_seconds = 10.0
        self.use_mock_tts = use_mock_tts

        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f).get("voice", {})
                self.voice_enabled = bool(cfg.get("voice_enabled", self.voice_enabled))
                self.voice_cooldown_seconds = float(cfg.get("voice_cooldown_seconds", self.voice_cooldown_seconds))
                self.repeat_error_after_seconds = float(cfg.get("repeat_error_after_seconds", self.repeat_error_after_seconds))

        self.last_spoken_message: Optional[str] = None
        self.last_spoken_category: Optional[str] = None
        self.last_spoken_time: float = -999.0

        # Spoken history for verification/testing
        self.spoken_history: List[Tuple[float, str, str]] = []  # (timestamp, category, message)

        self._tts_engine = None
        if HAS_PYTTSX3 and not self.use_mock_tts:
            try:
                self._tts_engine = pyttsx3.init()
                self._tts_engine.setProperty('rate', 160)
            except Exception:
                self._tts_engine = None

    def speak(
        self,
        message: str,
        category: str = "INFO",
        timestamp: Optional[float] = None,
        force: bool = False
    ) -> bool:
        """
        Triggers text-to-speech if voice is enabled and cooldown rules pass.
        Returns True if spoken, False if suppressed.
        """
        if not self.voice_enabled:
            return False

        current_time = timestamp if timestamp is not None else time.time()
        elapsed = current_time - self.last_spoken_time

        # Cooldown & De-duplication check
        if not force and message == self.last_spoken_message:
            cooldown = self.repeat_error_after_seconds if category in ("ERROR", "WARNING") else self.voice_cooldown_seconds
            if elapsed < cooldown:
                return False  # Suppress repeat message

        if not force and elapsed < self.voice_cooldown_seconds and category not in ("ERROR", "COMPLETE"):
            return False  # Rate limit general prompts

        self.last_spoken_message = message
        self.last_spoken_category = category
        self.last_spoken_time = current_time

        self.spoken_history.append((current_time, category, message))

        # Perform TTS output if engine available
        if self._tts_engine and not self.use_mock_tts:
            def _say():
                try:
                    self._tts_engine.say(message)
                    self._tts_engine.runAndWait()
                except Exception:
                    pass
            threading.Thread(target=_say, daemon=True).start()

        return True

    def alert(self, message: str, category: str = "ERROR", timestamp: Optional[float] = None) -> bool:
        """Convenience method for sending categorized alerts."""
        return self.speak(message, category=category, timestamp=timestamp)

    def speak_state_update(self, update: StateUpdate) -> bool:
        """
        Evaluates a StateUpdate and triggers appropriate rate-limited voice alerts.
        """
        if update.status == "STEP_COMPLETED":
            return self.speak(f"Step complete. Next: {update.message}", category="STEP", timestamp=update.timestamp)
        elif update.status == "COMPLETE":
            return self.speak(self.DEFAULT_ERROR_MESSAGES["COMPLETE"], category="COMPLETE", timestamp=update.timestamp, force=True)
        elif update.status == "RECOVERY":
            return self.speak(self.DEFAULT_ERROR_MESSAGES["RECOVERY"], category="INFO", timestamp=update.timestamp)
        elif update.status == "ERROR" and update.error_type:
            msg = self.DEFAULT_ERROR_MESSAGES.get(update.error_type, update.message)
            return self.speak(msg, category="ERROR", timestamp=update.timestamp)
        elif update.status == "WAITING" and update.error_type == "INACTION":
            return self.speak(self.DEFAULT_ERROR_MESSAGES["INACTION"], category="WARNING", timestamp=update.timestamp)
        elif update.status == "WAITING" and update.error_type == "PERCEPTION_UNCERTAIN":
            return self.speak(self.DEFAULT_ERROR_MESSAGES["PERCEPTION_UNCERTAIN"], category="WARNING", timestamp=update.timestamp)
        elif update.status == "ACTIVE" and update.transitioned:
            return self.speak(update.message, category="STEP", timestamp=update.timestamp)

        return False

    def reset(self) -> None:
        """Resets rate-limiting memory."""
        self.last_spoken_message = None
        self.last_spoken_category = None
        self.last_spoken_time = -999.0
        self.spoken_history.clear()
