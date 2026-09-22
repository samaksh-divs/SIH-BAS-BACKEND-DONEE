"""
Live GUI Launcher — Real Camera Mode
Launches the full ExperimentGUI (src/gui.py) with real webcam + real Person 1 models.

Usage:
    python scripts/run_gui_live.py

Steps:
    1. Window opens → select LIVE_CAMERA in the dropdown (default is REPLAY)
    2. Click "Start Camera"
    3. Confirm CAMERA: CONNECTED and Person 1 AI: CONNECTED
    4. Click "START EXPERIMENT"
    5. Perform experiment steps in front of camera
"""
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tkinter as tk
from src.gui import ExperimentGUI


def main():
    print("=" * 60)
    print("SIH EXPERIMENT 3 — LIVE GUI LAUNCHER")
    print("=" * 60)
    print("  Models dir   :", os.path.abspath("models"))
    print("  Detector     :", "models/best.pt")
    print("  Pose model   :", "models/yolo26n-pose.pt")
    print("  Recordings   :", os.path.abspath("recordings"))
    print("  Logs         :", os.path.abspath("logs"))
    print()
    print("  >> In the GUI: select LIVE_CAMERA, click Start Camera,")
    print("     then click START EXPERIMENT when camera is connected.")
    print("=" * 60)

    root = tk.Tk()
    # use_mock_tts=False → real voice alerts if TTS is available
    # Set use_mock_tts=True if you don't have a TTS engine installed
    gui = ExperimentGUI(root, use_mock_tts=False)
    root.mainloop()


if __name__ == "__main__":
    main()
