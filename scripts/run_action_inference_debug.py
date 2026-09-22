"""
Debug script to run Action Inference over person1_visual_output.jsonl
and summarize observed action hypotheses over time.
"""
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.input_adapter import Person1Adapter
from src.action_inference import ActionInferenceEngine


def run_debug_replay(dataset_path: str = "data/person1_visual_output.jsonl"):
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' does not exist.")
        return

    print("=========================================================")
    print("      SIH EXPERIMENT 3 - ACTION INFERENCE DEBUG REPLAY   ")
    print("=========================================================")
    print(f"Dataset: {dataset_path}\n")

    engine = ActionInferenceEngine(config_path="config/thresholds.json")

    action_counts = {}
    prev_action = None
    transitions = []

    total_frames = 0

    for frame in Person1Adapter.stream_file(dataset_path):
        total_frames += 1
        obs = engine.process_frame(frame)

        action_counts[obs.action] = action_counts.get(obs.action, 0) + 1

        if obs.action != prev_action:
            transitions.append((frame.frame, frame.timestamp, obs.action, obs.confidence_level, obs.reason))
            prev_action = obs.action

    print(f"Processed {total_frames} frames successfully.\n")
    print("--- DETECTED ACTION HYPOTHESES SUMMARY ---")
    for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_frames) * 100.0
        print(f"  {action:<28}: {count:>4} frames ({percentage:>5.1f}%)")

    print("\n--- SAMPLE ACTION TRANSITIONS (First 20 transitions) ---")
    for frame_num, ts, act, conf_lvl, reason in transitions[:20]:
        print(f"  Frame {frame_num:>4} (t={ts:>6.2f}s) -> [{act:<25}] ({conf_lvl}) | {reason}")

    print("\nDebug replay finished.")


if __name__ == "__main__":
    dataset_file = sys.argv[1] if len(sys.argv) > 1 else "data/person1_visual_output.jsonl"
    run_debug_replay(dataset_file)
