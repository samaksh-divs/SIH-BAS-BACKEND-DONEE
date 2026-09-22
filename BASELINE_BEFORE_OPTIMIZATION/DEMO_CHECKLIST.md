# SIH EXPERIMENT 3 — LIVE DEMO OPERATOR CHECKLIST

This document provides a step-by-step checklist for operators running live demonstrations and SIH acceptance evaluations of the Person 2 Autonomous Experiment Tracking System.

---

## Pre-Requisites & System Setup

- Ensure webcam device is connected to USB port.
- Verify model files exist:
  - `models/best.pt` (Person 1 detector model)
  - `models/yolo26n-pose.pt` (Person 1 pose keypoint model)
- Ensure speakers/audio output is enabled for offline voice guidance alerts.
- Check network connectivity (if LAN stream demonstration is desired).

---

## 18-Step Operator Execution Checklist

- [ ] **Step 1: Start Application**
  Launch monitoring application:
  ```bash
  python scripts/run_gui_demo.py
  ```

- [ ] **Step 2: Confirm Camera Connection**
  Check the GUI top-right status banner: `CAMERA: CONNECTED`. Confirm live video feed appears on the left canvas.

- [ ] **Step 3: Confirm Person 1 Detector**
  Verify `PERSON 1 AI: CONNECTED`. Confirm object bounding boxes (`white_container`, `red_box`, `yellow_box`, `plant`, `spray_bottle`) are rendered on video frames.

- [ ] **Step 4: Confirm Pose Estimation**
  Verify human wrist keypoints (`left_wrist`, `right_wrist`) are detected and highlighted on video frames.

- [ ] **Step 5: Confirm Person 2 Application Logic**
  Verify `PERSON 2: STOPPED` (Ready for start) and step progress reads `1 / 15 (S01)`.

- [ ] **Step 6: Confirm Voice Guidance**
  Ensure audio volume is audible. The voice manager uses offline TTS (pyttsx3 / Windows SAPI5).

- [ ] **Step 7: Confirm Local Recording Setup**
  Verify `RECORDING: ON` when experiment starts, targeting `recordings/experiment_YYYYMMDD_HHMMSS.mp4`.

- [ ] **Step 8: Confirm IP Video Stream URL**
  Verify `IP STREAM: http://<HOST_LAN_IP>:8554/stream` is active. Open a browser on an external LAN device (phone/laptop) to confirm stream playback.

- [ ] **Step 9: Start Experiment**
  Click **"START EXPERIMENT"** button in GUI. Confirm system status changes to `EXPERIMENT ACTIVE` and recording begins.

- [ ] **Step 10: Perform Step 1 (OPEN_WHITE_BOX)**
  Operator opens the white container. Confirm step completes automatically and transitions to `S02`.

- [ ] **Step 11: Observe Next-Step Guidance**
  Verify voice alert speaks: *"Step complete. Next: Retrieve the red box."* and GUI displays `Next Step: S02 RETRIEVE_RED_BOX`.

- [ ] **Step 12: Introduce a Procedural Error (Simulated Skipped / Out-of-Order Step)**
  Operator intentionally skips retrieving the red box and directly opens the yellow box out of order.

- [ ] **Step 13: Observe Warning Alert**
  Verify GUI displays `STATUS: ERROR` with red highlight and voice alert warns: *"Out of sequence step detected. Expected: Retrieve the red box."*

- [ ] **Step 14: Continue / Recover Experiment**
  Operator returns to the red box or continues expected flow. Confirm system status updates to `RECOVERY` and auto-resynchronizes state tracking.

- [ ] **Step 15: Finish Experiment (S15 CLOSE_WHITE_BOX)**
  Operator completes final step by closing the white box.

- [ ] **Step 16: Confirm COMPLETE Status**
  Verify system displays `STATUS: COMPLETE` and speaks: *"Experiment complete. All steps finished successfully."*

- [ ] **Step 17: Show Saved Video File**
  Open the recorded file under `recordings/` using OpenCV/VLC and confirm video integrity and duration.

- [ ] **Step 18: Show Structured Event Log**
  Open `logs/experiment_log_YYYYMMDD_HHMMSS.jsonl` and highlight `EXPERIMENT_STARTED`, `STEP_COMPLETED` (with explicit `completed_state` and `next_state`), and `EXPERIMENT_COMPLETED` event lines.

---

## Verification Summary Command

To run the complete automated test suite and generate the final acceptance report at any time:

```bash
# Run unit tests
python -m unittest discover tests

# Verify IP streaming across localhost and LAN
python scripts/verify_streaming.py

# Verify spray interaction resilience
python scripts/verify_spray_interaction.py

# Profile pipeline performance & latency
python scripts/profile_pipeline.py

# Generate final SIH acceptance matrix & report
python scripts/generate_sih_acceptance_report.py
```
