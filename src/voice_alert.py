"""
Offline Voice Alert Manager
Provides rate-limited offline text-to-speech alerts using pyttsx3 (with fallback for mock/test environments).
"""
import json
import os
import threading
import time
import queue
from typing import Dict, List, Optional, Tuple, Any

from src.state_machine import StateUpdate

try:
    import pyttsx3
    HAS_PYTTSX3 = True
except ImportError:
    HAS_PYTTSX3 = False


def _tts_worker_loop(q: queue.Queue):
    if not HAS_PYTTSX3:
        return
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)
    except Exception:
        return
    
    while True:
        msg = q.get()
        if msg is None:
            break
        try:
            engine.say(msg)
            engine.runAndWait()
        except Exception:
            pass
        q.task_done()


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
        "COMPLETE": "Congratulations! Experiment complete. Well done!"
    }

    # Human-readable step instructions for voice guidance
    STEP_INSTRUCTIONS = {
        "OPEN_WHITE_BOX":           "Step 1. Open the white container.",
        "RETRIEVE_RED_BOX":         "Step 2. Retrieve the red box from the white container.",
        "RETRIEVE_YELLOW_BOX":      "Step 3. Retrieve the yellow box from the white container.",
        "OPEN_RED_BOX":             "Step 4. Open the red box.",
        "PLANT_TO_WORKPLACE":       "Step 5. Move the plant to the workplace.",
        "OPEN_YELLOW_BOX":          "Step 6. Open the yellow box.",
        "SPRAY_TO_WORKPLACE":       "Step 7. Place the spray bottle at the workplace.",
        "PICK_SPRAY":               "Step 8. Pick up the spray bottle.",
        "SPRAY_PLANT":              "Step 9. Spray the plant.",
        "SPRAY_TO_WORKPLACE_AGAIN": "Step 10. Return the spray bottle to the workplace.",
        "PLANT_TO_RED_BOX":         "Step 11. Move the plant to the red box.",
        "SPRAY_TO_YELLOW_BOX":      "Step 12. Move the spray bottle to the yellow box.",
        "RED_BOX_TO_WHITE_BOX":     "Step 13. Move the red box into the white container.",
        "YELLOW_BOX_TO_WHITE_BOX":  "Step 14. Move the yellow box into the white container.",
        "CLOSE_WHITE_BOX":          "Step 15. Close the white container.",
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

        # TTS engine queue to prevent Windows COM deadlocks
        self._tts_enabled = HAS_PYTTSX3 and not self.use_mock_tts
        self._tts_queue = queue.Queue()
        if self._tts_enabled:
            self._tts_thread = threading.Thread(target=_tts_worker_loop, args=(self._tts_queue,), daemon=True)
            self._tts_thread.start()

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

        # Send message to the dedicated TTS thread
        if self._tts_enabled:
            self._tts_queue.put(message)

        return True

    def alert(self, message: str, category: str = "ERROR", timestamp: Optional[float] = None) -> bool:
        """Convenience method for sending categorized alerts."""
        return self.speak(message, category=category, timestamp=timestamp)

    def speak_experiment_start(self) -> None:
        """Announces experiment start and guides person to Step 1."""
        self.speak(
            "Let's start the experiment! "
            "Step 1. Open the white container.",
            category="STEP",
            force=True
        )

    def speak_state_update(self, update: StateUpdate) -> bool:
        """
        Evaluates a StateUpdate and triggers appropriate rate-limited voice alerts.
        """
        if update.transitioned:
            # Announce success and instruct the NEXT step
            next_action = update.expected_action or ""
            next_instruction = self.STEP_INSTRUCTIONS.get(next_action, "")
            if next_instruction:
                msg = f"Step complete! {next_instruction}"
            else:
                msg = "Step complete!"
            return self.speak(msg, category="STEP", timestamp=update.timestamp, force=True)
        elif update.status == "COMPLETE":
            return self.speak(self.DEFAULT_ERROR_MESSAGES["COMPLETE"], category="COMPLETE", timestamp=update.timestamp, force=True)
        elif update.status == "RECOVERY":
            # Guide person to the recovered step
            recovered_action = update.expected_action or ""
            recovered_instruction = self.STEP_INSTRUCTIONS.get(recovered_action, "")
            msg = self.DEFAULT_ERROR_MESSAGES["RECOVERY"]
            if recovered_instruction:
                msg = f"{msg} Please continue: {recovered_instruction}"
            return self.speak(msg, category="INFO", timestamp=update.timestamp)
        elif update.status == "ERROR" and update.error_type:
            # MUTE OUT OF SEQUENCE ERRORS - NEVER GET STUCK YELLING
            if update.error_type in ("OUT_OF_SEQUENCE", "SKIPPED_STEP", "WRONG_OBJECT"):
                return False

            msg = self.DEFAULT_ERROR_MESSAGES.get(update.error_type, update.message)
            return self.speak(msg, category="ERROR", timestamp=update.timestamp)
        elif update.status == "WAITING" and update.error_type == "INACTION":
            # Remind with the specific step instruction
            action_instruction = self.STEP_INSTRUCTIONS.get(update.expected_action, "")
            if action_instruction:
                msg = f"Please continue. {action_instruction}"
            else:
                msg = self.DEFAULT_ERROR_MESSAGES["INACTION"]
            return self.speak(msg, category="WARNING", timestamp=update.timestamp)
        elif update.status == "WAITING" and update.error_type == "PERCEPTION_UNCERTAIN":
            return False  # Mute uncertain errors to avoid spam
        
        return False

    def reset(self) -> None:
        """Resets rate-limiting memory."""
        self.last_spoken_message = None
        self.last_spoken_category = None
        self.last_spoken_time = -999.0
        self.spoken_history.clear()
