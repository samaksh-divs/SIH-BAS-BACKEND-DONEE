"""
Synthetic Demo Script for State Machine & Error Detection
Demonstrates:
1. Normal complete sequence S01 -> S16
2. Skipped-step error scenario
3. Out-of-sequence error scenario
"""
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.action_inference import ObservedAction
from src.state_machine import ExperimentStateMachine


def make_synthetic_action(action_name: str, frame: int, timestamp: float, confidence: float = 0.90) -> ObservedAction:
    return ObservedAction(
        action=action_name,
        confidence=confidence,
        confidence_level="HIGH" if confidence >= 0.85 else "MEDIUM",
        evidence=[f"Synthetic evidence for {action_name}"],
        start_frame=frame,
        end_frame=frame,
        timestamp=timestamp,
        involved_objects=[],
        wrist_side="right",
        reason=f"Synthetic action observation of {action_name}"
    )


def print_step_header(title: str):
    print("\n" + "=" * 80)
    print(f"   {title}")
    print("=" * 80)
    print(f"{'Frame':<7} {'Expected':<22} {'Observed':<22} {'State':<7} {'Status':<12} {'Error':<18} {'Message'}")
    print("-" * 80)


def run_normal_sequence_demo():
    print_step_header("DEMO 1: NORMAL EXPERIMENT SEQUENCE (S01 -> S16)")
    sm = ExperimentStateMachine()

    sequence_actions = [
        "OPEN_WHITE_BOX", "RETRIEVE_RED_BOX", "RETRIEVE_YELLOW_BOX",
        "OPEN_RED_BOX", "PLANT_TO_WORKPLACE", "OPEN_YELLOW_BOX",
        "SPRAY_TO_WORKPLACE", "PICK_SPRAY", "SPRAY_PLANT",
        "SPRAY_TO_WORKPLACE_AGAIN", "PLANT_TO_RED_BOX", "SPRAY_TO_YELLOW_BOX",
        "RED_BOX_TO_WHITE_BOX", "YELLOW_BOX_TO_WHITE_BOX", "CLOSE_WHITE_BOX"
    ]

    frame = 1
    timestamp = 0.0

    for action_name in sequence_actions:
        obs = make_synthetic_action(action_name, frame, timestamp)
        for _ in range(3):
            timestamp += 0.1
            up = sm.update(obs, frame=frame, timestamp=timestamp)
            err_str = up.error_type if up.error_type else "-"
            print(f"{frame:<7} {up.expected_action:<22} {up.observed_action:<22} {up.current_state_id:<7} {up.status:<12} {err_str:<18} {up.message}")
            frame += 1


def run_skipped_step_demo():
    print_step_header("DEMO 2: SKIPPED STEP SCENARIO (Expected S08 PICK_SPRAY -> Observed S09 SPRAY_PLANT)")
    sm = ExperimentStateMachine()
    sm.resync_to_step(8)  # Fast-forward to S08

    frame = 100
    timestamp = 10.0

    # Operator jumps straight to spraying plant without picking spray
    skipped_obs = make_synthetic_action("SPRAY_PLANT", frame, timestamp)

    for i in range(5):
        timestamp += 0.1
        up = sm.update(skipped_obs, frame=frame, timestamp=timestamp)
        err_str = up.error_type if up.error_type else "-"
        print(f"{frame:<7} {up.expected_action:<22} {up.observed_action:<22} {up.current_state_id:<7} {up.status:<12} {err_str:<18} {up.message}")
        frame += 1


def run_out_of_sequence_demo():
    print_step_header("DEMO 3: OUT-OF-SEQUENCE SCENARIO (Expected S02 RETRIEVE_RED_BOX -> Observed S03 RETRIEVE_YELLOW_BOX)")
    sm = ExperimentStateMachine()
    sm.resync_to_step(2)  # S02 RETRIEVE_RED_BOX

    frame = 50
    timestamp = 5.0

    bad_obs = make_synthetic_action("RETRIEVE_YELLOW_BOX", frame, timestamp)

    for i in range(3):
        timestamp += 0.1
        up = sm.update(bad_obs, frame=frame, timestamp=timestamp)
        err_str = up.error_type if up.error_type else "-"
        print(f"{frame:<7} {up.expected_action:<22} {up.observed_action:<22} {up.current_state_id:<7} {up.status:<12} {err_str:<18} {up.message}")
        frame += 1


if __name__ == "__main__":
    run_normal_sequence_demo()
    run_skipped_step_demo()
    run_out_of_sequence_demo()
