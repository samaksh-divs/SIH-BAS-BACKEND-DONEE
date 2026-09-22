"""
Synthetic GUI Demo Script (Part I)
Launches the Monitoring GUI with synthetic playback demonstrating step transitions, alerts,
skipped step error, recovery, and completion.
"""
import os
import sys
import tkinter as tk
import threading
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.gui import ExperimentGUI
from src.action_inference import ObservedAction


def make_obs(action_name: str, frame: int, timestamp: float) -> ObservedAction:
    return ObservedAction(
        action=action_name,
        confidence=0.92,
        confidence_level="HIGH",
        evidence=[f"Synthetic evidence for {action_name}"],
        start_frame=frame,
        end_frame=frame,
        timestamp=timestamp,
        involved_objects=[],
        wrist_side="right",
        reason=f"Synthetic demo action {action_name}"
    )


def run_synthetic_gui_playback(gui: ExperimentGUI):
    """Feeds synthetic sequence frames into GUI pipeline to demonstrate visual updates."""
    time.sleep(1.0)

    frame = 1
    timestamp = 0.0

    # 1. S01 OPEN_WHITE_BOX
    for _ in range(3):
        timestamp += 0.1
        gui.pipeline.process_frame({
            "frame": frame, "timestamp": timestamp, "objects": [{"class": "white_container", "bbox": [10, 10, 100, 100]}],
            "pose": {}, "hand_object_interaction": {},
            "interaction_signals": {"right_hand_near_white_container": True}
        })
        frame += 1
        time.sleep(0.3)

    # 2. S02 RETRIEVE_RED_BOX
    for _ in range(3):
        timestamp += 0.1
        gui.pipeline.process_frame({
            "frame": frame, "timestamp": timestamp,
            "objects": [{"class": "red_box", "bbox": [10 + frame*5, 10 + frame*5, 100, 100]}],
            "pose": {}, "hand_object_interaction": {},
            "interaction_signals": {"right_hand_near_red_box": True}
        })
        frame += 1
        time.sleep(0.3)

    # 3. Fast-forward to S08 & trigger skipped step error
    gui.pipeline.state_machine.resync_to_step(8)

    for i in range(5):
        timestamp += 0.1
        gui.pipeline.process_frame({
            "frame": frame, "timestamp": timestamp,
            "objects": [{"class": "spray_bottle", "bbox": [50, 50, 80, 80]}, {"class": "plant", "bbox": [200, 200, 250, 250]}],
            "pose": {}, "hand_object_interaction": {},
            "interaction_signals": {"right_hand_near_spray_bottle": True, "right_hand_near_plant": True}
        })
        frame += 1
        time.sleep(0.4)

    # 4. Finish sequence
    gui.pipeline.state_machine.resync_to_step(16)
    gui.pipeline.process_frame({
        "frame": frame, "timestamp": timestamp + 0.1, "objects": [], "pose": {},
        "hand_object_interaction": {}, "interaction_signals": {}
    })


def main():
    root = tk.Tk()
    gui = ExperimentGUI(root, use_mock_tts=True)

    # Start synthetic playback thread
    threading.Thread(target=run_synthetic_gui_playback, args=(gui,), daemon=True).start()

    root.mainloop()


if __name__ == "__main__":
    main()
