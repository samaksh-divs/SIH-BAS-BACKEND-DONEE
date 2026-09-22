"""
Integration Demo for Event Logging & Voice Alerts (Part H)
Demonstrates end-to-end integration of State Machine -> Event Logger + Voice Alert Manager.
"""
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.action_inference import ObservedAction
from src.state_machine import ExperimentStateMachine
from src.event_logger import EventLogger
from src.voice_alert import VoiceAlertManager


def make_obs(action_name: str, frame: int, timestamp: float) -> ObservedAction:
    return ObservedAction(
        action=action_name,
        confidence=0.90,
        confidence_level="HIGH",
        evidence=[f"Evidence for {action_name}"],
        start_frame=frame,
        end_frame=frame,
        timestamp=timestamp,
        involved_objects=[],
        wrist_side="right",
        reason=f"Observed {action_name}"
    )


def run_demo():
    print("=========================================================================")
    print("      SIH EXPERIMENT 3 - EVENT LOGGING & VOICE ALERTS INTEGRATION DEMO   ")
    print("=========================================================================\n")

    sm = ExperimentStateMachine()
    logger = EventLogger(log_dir="logs", custom_filename="demo_integration.jsonl")
    voice = VoiceAlertManager(use_mock_tts=True)

    frame = 1
    timestamp = 0.0

    # 1. START & S01 OPEN_WHITE_BOX
    print("--- 1. STARTING S01 (OPEN_WHITE_BOX) ---")
    s01_obs = make_obs("OPEN_WHITE_BOX", frame, timestamp)
    for _ in range(3):
        timestamp += 0.1
        up = sm.update(s01_obs, frame=frame, timestamp=timestamp)
        logger.log_state_update(up)
        voice.speak_state_update(up)
        frame += 1

    # 2. S02 RETRIEVE_RED_BOX
    print("\n--- 2. STARTING S02 (RETRIEVE_RED_BOX) ---")
    s02_obs = make_obs("RETRIEVE_RED_BOX", frame, timestamp)
    for _ in range(3):
        timestamp += 0.1
        up = sm.update(s02_obs, frame=frame, timestamp=timestamp)
        logger.log_state_update(up)
        voice.speak_state_update(up)
        frame += 1

    # Fast-forward to S08
    sm.resync_to_step(8)

    # 3. SKIPPED STEP ERROR & REPEATED SAME ERROR AT S08
    print("\n--- 3. SKIPPED STEP ERROR AT S08 (Expected S08 PICK_SPRAY -> Observed S09 SPRAY_PLANT) ---")
    skipped_obs = make_obs("SPRAY_PLANT", frame, timestamp)
    for i in range(5):
        timestamp += 0.1
        up = sm.update(skipped_obs, frame=frame, timestamp=timestamp)
        logger.log_state_update(up)
        voice.speak_state_update(up)
        frame += 1

    # 4. FAST FORWARD TO S16 (COMPLETE)
    print("\n--- 4. FINISHING EXPERIMENT (S16 COMPLETE) ---")
    sm.resync_to_step(16)
    complete_obs = make_obs("COMPLETE", frame, timestamp)
    up = sm.update(complete_obs, frame=frame, timestamp=timestamp)
    logger.log_state_update(up)
    voice.speak_state_update(up)

    logger.close()

    print("\n" + "=" * 80)
    print("--- GENERATED LOG FILE LOCATION ---")
    print(f"Log File Path: {os.path.abspath(logger.filepath)}")
    print("=" * 80)

    print("\n--- SPOKEN VOICE MESSAGES HISTORY ---")
    for ts, cat, msg in voice.spoken_history:
        print(f"  [{ts:>5.2f}s] ({cat:<8}): \"{msg}\"")

    print(f"\nTotal Logged Events: {logger.logged_events_count}")
    print(f"Total Spoken Alerts: {len(voice.spoken_history)}")
    print("Demo execution finished successfully.\n")


if __name__ == "__main__":
    run_demo()
