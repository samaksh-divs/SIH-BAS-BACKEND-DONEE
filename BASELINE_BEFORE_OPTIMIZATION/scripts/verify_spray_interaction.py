"""
Spray Interaction Validation Script (Part F).
Validates spray bottle steps:
S07: SPRAY_TO_WORKPLACE
S08: PICK_SPRAY
S09: SPRAY_PLANT
S10: SPRAY_TO_WORKPLACE_AGAIN
S12: SPRAY_TO_YELLOW_BOX
Verifies occlusion resilience, wrist availability, hand proximity signals, and state progression.
"""
import os
import sys
import json

sys.path.insert(0, os.path.abspath('.'))
from src.pipeline import ExperimentPipeline

def verify_spray_sequence():
    print("========================================")
    print("PART F — SPRAY INTERACTION VALIDATION   ")
    print("========================================")
    
    pipeline = ExperimentPipeline(log_dir="logs", use_mock_tts=True)
    
    # Fast-forward state machine to S06 (ready for S07)
    pipeline.state_machine.resync_to_step(7) # S07 SPRAY_TO_WORKPLACE
    
    print(f"Resynced to initial state: {pipeline.state_machine.current_state.state_id} (Expected interaction: {pipeline.state_machine.current_state.expected_interaction})")
    
    results = []

    # 1. Test S07: SPRAY_TO_WORKPLACE (moving spray_bottle from yellow box to workplace)
    print("\n[Testing S07: SPRAY_TO_WORKPLACE]")
    frame_num = 1
    timestamp = 0.1
    for step in range(5):
        timestamp += 0.1
        frame_num += 1
        rec = {
            "frame": frame_num, "timestamp": timestamp,
            "objects": [{"class": "spray_bottle", "track_id": 5, "confidence": 0.9, "bbox": [100 + step*10, 100 + step*10, 150 + step*10, 150 + step*10]}],
            "pose": {"right_wrist": {"x": 125.0 + step*10, "y": 125.0 + step*10, "confidence": 0.88}},
            "hand_object_interaction": {},
            "interaction_signals": {"right_hand_near_spray_bottle": True}
        }
        up, norm_f, obs = pipeline.process_frame(rec)
        if up.transitioned:
            print(f"  Frame {frame_num}: Transited to {up.current_state_id} ({up.observed_action})")
            results.append(("S07", True, up.current_state_id))
            break
            
    # 2. Test S08: PICK_SPRAY (Lifting spray bottle vertically)
    print("\n[Testing S08: PICK_SPRAY]")
    for step in range(5):
        timestamp += 0.1
        frame_num += 1
        rec = {
            "frame": frame_num, "timestamp": timestamp,
            "objects": [{"class": "spray_bottle", "track_id": 5, "confidence": 0.9, "bbox": [150, 150, 200, 200]}],
            "pose": {"right_wrist": {"x": 175.0, "y": 175.0 - (step * 20.0), "confidence": 0.88}}, # Lifting UP
            "hand_object_interaction": {},
            "interaction_signals": {"right_hand_near_spray_bottle": True}
        }
        up, norm_f, obs = pipeline.process_frame(rec)
        if up.transitioned:
            print(f"  Frame {frame_num}: Transited to {up.current_state_id} ({up.observed_action})")
            results.append(("S08", True, up.current_state_id))
            break

    # 3. Test S09: SPRAY_PLANT (Hand near spray + hand near plant) with 2-frame occlusion gap
    print("\n[Testing S09: SPRAY_PLANT with Occlusion Tolerance]")
    for step in range(8):
        timestamp += 0.1
        frame_num += 1
        
        # Simulate 2-frame temporary occlusion of spray_bottle at step 2 & 3
        is_occluded = (step in [2, 3])
        objs = [] if is_occluded else [
            {"class": "spray_bottle", "track_id": 5, "confidence": 0.9, "bbox": [150, 150, 200, 200]},
            {"class": "plant", "track_id": 4, "confidence": 0.85, "bbox": [220, 150, 280, 200]}
        ]
        
        signals = {"right_hand_near_plant": True}
        if not is_occluded:
            signals["right_hand_near_spray_bottle"] = True

        rec = {
            "frame": frame_num, "timestamp": timestamp,
            "objects": objs,
            "pose": {"right_wrist": {"x": 175.0, "y": 160.0, "confidence": 0.88}},
            "hand_object_interaction": {},
            "interaction_signals": signals
        }
        up, norm_f, obs = pipeline.process_frame(rec)
        if is_occluded:
            print(f"  Frame {frame_num} (OCCLUDED): Action={obs.action}, Status={up.status}")
        if up.transitioned:
            print(f"  Frame {frame_num}: Transited to {up.current_state_id} ({up.observed_action})")
            results.append(("S09", True, up.current_state_id))
            break

    # 4. Test S10: SPRAY_TO_WORKPLACE_AGAIN (Lowering spray bottle back)
    print("\n[Testing S10: SPRAY_TO_WORKPLACE_AGAIN]")
    for step in range(5):
        timestamp += 0.1
        frame_num += 1
        rec = {
            "frame": frame_num, "timestamp": timestamp,
            "objects": [{"class": "spray_bottle", "track_id": 5, "confidence": 0.9, "bbox": [150, 150, 200, 200]}],
            "pose": {"right_wrist": {"x": 175.0, "y": 160.0 + (step * 20.0), "confidence": 0.88}}, # Lowering DOWN
            "hand_object_interaction": {},
            "interaction_signals": {"right_hand_near_spray_bottle": True}
        }
        up, norm_f, obs = pipeline.process_frame(rec)
        if up.transitioned:
            print(f"  Frame {frame_num}: Transited to {up.current_state_id} ({up.observed_action})")
            results.append(("S10", True, up.current_state_id))
            break

    # Resync to S12: SPRAY_TO_YELLOW_BOX
    pipeline.state_machine.resync_to_step(12)
    print(f"\nResynced to state: {pipeline.state_machine.current_state.state_id} (Expected interaction: {pipeline.state_machine.current_state.expected_interaction})")

    # 5. Test S12: SPRAY_TO_YELLOW_BOX (Returning spray bottle to yellow box)
    print("\n[Testing S12: SPRAY_TO_YELLOW_BOX]")
    for step in range(5):
        timestamp += 0.1
        frame_num += 1
        rec = {
            "frame": frame_num, "timestamp": timestamp,
            "objects": [
                {"class": "spray_bottle", "track_id": 5, "confidence": 0.9, "bbox": [300 + step*10, 400 + step*10, 350 + step*10, 450 + step*10]},
                {"class": "yellow_box", "track_id": 3, "confidence": 0.85, "bbox": [300, 400, 400, 500]}
            ],
            "pose": {"right_wrist": {"x": 320.0 + step*10, "y": 420.0 + step*10, "confidence": 0.88}},
            "hand_object_interaction": {},
            "interaction_signals": {"right_hand_near_spray_bottle": True, "right_hand_near_yellow_box": True}
        }
        up, norm_f, obs = pipeline.process_frame(rec)
        if up.transitioned:
            print(f"  Frame {frame_num}: Transited to {up.current_state_id} ({up.observed_action})")
            results.append(("S12", True, up.current_state_id))
            break

    pipeline.close()
    
    print("\n========================================")
    print("SPRAY INTERACTION VERIFICATION RESULTS:")
    for step_id, passed, target in results:
        print(f"  {step_id}: {'VERIFIED' if passed else 'FAILED'} -> next state {target}")
    print("========================================")
    return len(results) >= 5

if __name__ == "__main__":
    verify_spray_sequence()
