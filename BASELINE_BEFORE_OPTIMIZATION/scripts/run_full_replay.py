"""
Full Replay Pipeline Execution Script (Part J)
Runs real sequential replay against data/person1_visual_output.jsonl (1,860 frames)
and reports comprehensive execution statistics.
"""
import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import ExperimentPipeline
from src.input_adapter import Person1Adapter


def run_full_replay_headless(dataset_path: str = "data/person1_visual_output.jsonl"):
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file '{dataset_path}' not found.")
        return

    print("=========================================================================")
    print("      SIH EXPERIMENT 3 - FULL REPLAY PIPELINE EXECUTION (HEADLESS)       ")
    print("=========================================================================")
    print(f"Dataset File: {dataset_path}\n")

    pipeline = ExperimentPipeline(log_dir="logs", use_mock_tts=True)

    start_time = time.time()
    frames_processed = 0
    action_hypotheses_counts = {}
    state_transitions = []
    errors_detected = []
    uncertain_frames_count = 0

    for frame in Person1Adapter.stream_file(dataset_path):
        frames_processed += 1
        line_str = json_record = {
            "frame": frame.frame,
            "timestamp": frame.timestamp,
            "objects": [{"class": obj.class_name, "track_id": obj.track_id, "confidence": obj.confidence, "bbox": list(obj.bbox)} for obj in frame.objects],
            "pose": {
                "person_track_id": frame.pose.person_track_id,
                "left_wrist": {"x": frame.pose.left_wrist.x, "y": frame.pose.left_wrist.y, "confidence": frame.pose.left_wrist.confidence},
                "right_wrist": {"x": frame.pose.right_wrist.x, "y": frame.pose.right_wrist.y, "confidence": frame.pose.right_wrist.confidence}
            },
            "hand_object_interaction": {
                "left": {k: {"near": v.near, "stable": v.stable, "track_id": v.track_id, "distance_px": v.distance_px} for k, v in frame.hand_object_interaction.left.items()},
                "right": {k: {"near": v.near, "stable": v.stable, "track_id": v.track_id, "distance_px": v.distance_px} for k, v in frame.hand_object_interaction.right.items()}
            },
            "interaction_signals": frame.interaction_signals
        }

        update, norm_frame, obs_action = pipeline.process_frame(json_record)

        act = obs_action.action
        action_hypotheses_counts[act] = action_hypotheses_counts.get(act, 0) + 1

        if obs_action.confidence_level == "UNCERTAIN" or act == "UNCERTAIN":
            uncertain_frames_count += 1

        if update.transitioned:
            state_transitions.append((frame.frame, frame.timestamp, update.previous_state, update.current_state_id, update.expected_action))

        if update.status == "ERROR" and update.error_type:
            errors_detected.append((frame.frame, frame.timestamp, update.error_type, update.expected_action, update.observed_action, update.message))

    elapsed_time = time.time() - start_time
    pipeline.close()

    print(f"--- REPLAY SUMMARY ---")
    print(f"Total Frames Processed : {frames_processed}")
    print(f"Elapsed Time           : {elapsed_time:.3f} seconds ({frames_processed / elapsed_time:.1f} FPS)")
    print(f"Final State            : Step {pipeline.state_machine.current_state.step_number} ({pipeline.state_machine.current_state.state_id})")
    print(f"Final Status           : {pipeline.state_machine.status}")
    print(f"Uncertain Frames Count : {uncertain_frames_count} frames ({uncertain_frames_count / frames_processed * 100:.1f}%)")
    print(f"Event Log Saved To     : {os.path.abspath(pipeline.logger.filepath)}\n")

    print("--- OBSERVED ACTION HYPOTHESES ---")
    for act, cnt in sorted(action_hypotheses_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {act:<28}: {cnt:>4} frames ({cnt / frames_processed * 100:>5.1f}%)")

    print("\n--- STATE TRANSITIONS RECORDED ---")
    for f_num, ts, prev_s, new_s, exp_act in state_transitions:
        print(f"  Frame {f_num:>4} (t={ts:>6.2f}s) | Transition: {prev_s or 'START':<5} -> {new_s:<5} [{exp_act}]")

    print("\n--- PROCEDURAL ERRORS DETECTED ---")
    if not errors_detected:
        print("  None (Perfect sequence match or debounced).")
    else:
        for f_num, ts, err_t, exp_a, obs_a, msg in errors_detected[:10]:
            print(f"  Frame {f_num:>4} (t={ts:>6.2f}s) | Error: {err_t:<15} | Exp: {exp_a:<20} | Obs: {obs_a:<20} | {msg}")

    print("\nFull replay pipeline run finished.")


if __name__ == "__main__":
    dataset_file = sys.argv[1] if len(sys.argv) > 1 else "data/person1_visual_output.jsonl"
    run_full_replay_headless(dataset_file)
