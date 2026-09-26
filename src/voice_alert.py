"""
Offline Voice Alert Manager
Provides rate-limited offline text-to-speech alerts using pyttsx3 (with fallback for mock/test environments).
"""
import json
import os
import threading
import time
import queue
import winsound
from typing import Dict, List, Optional, Tuple, Any

from src.state_machine import StateUpdate

def _tts_worker_loop(q: queue.Queue):
    audio_dir = r"C:\Users\Divya\Downloads\experiment_audio_stepwise"
    while True:
        msg = q.get()
        if msg is None:
            break
        
        if msg.startswith("FILE:"):
            step_id = msg.split(":")[1]
            filepath = os.path.join(audio_dir, f"{step_id}.wav")
            if os.path.exists(filepath):
                try:
                    # Play sound synchronously (but inside this background thread)
                    winsound.PlaySound(filepath, winsound.SND_FILENAME)
                except Exception as e:
                    print(f"[Audio] Error playing {filepath}: {e}")
            else:
                print(f"[Audio] WARNING: Missing audio file {filepath}")
        
        q.task_done()


class VoiceAlertManager:
    """
    Plays custom pre-recorded WAV files for each step.
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
        self.use_mock_tts = use_mock_tts

        # TTS engine queue to prevent Windows deadlocks
        self._tts_enabled = not self.use_mock_tts
        self._tts_queue = queue.Queue()
        self._warning_played = False  # ensure WARNING.wav only fires once per error window
        if self._tts_enabled:
            self._tts_thread = threading.Thread(target=_tts_worker_loop, args=(self._tts_queue,), daemon=True)
            self._tts_thread.start()

    def speak_experiment_start(self) -> None:
        """Announces experiment start and guides person to Step 1."""
        if self._tts_enabled:
            self._tts_queue.put("FILE:S01")

    def speak_state_update(self, update: StateUpdate) -> bool:
        """
        Triggers custom audio file playback on step transition or scripted error.
        """
        if update.transitioned:
            # Reset warning flag so it can fire again on future runs
            self._warning_played = False
            if self._tts_enabled:
                next_id = "S" + str(update.current_step).zfill(2)
                if next_id != "S16":
                    self._tts_queue.put(f"FILE:{next_id}")
            return True

        # Play WARNING.wav exactly once when the scripted error fires
        if update.status == "ERROR" and update.error_type == "WRONG_OBJECT":
            if not self._warning_played and self._tts_enabled:
                self._warning_played = True
                self._tts_queue.put("FILE:WARNING")
            return True

        return False

    def reset(self) -> None:
        """Resets the player."""
        pass
