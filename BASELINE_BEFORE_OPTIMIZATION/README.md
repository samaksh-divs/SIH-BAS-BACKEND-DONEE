# SIH Experiment 3 – Autonomous Experiment Execution Logic (Person 2)

This repository contains the Person 2 application logic layer for autonomous experiment execution monitoring, state tracking, procedural error detection, event logging, offline voice alerts, and real-time GUI monitoring.

---

## Shared Downstream Architecture

Both **Replay Mode** (simulated perception stream) and **Live Camera Mode** feed through the **SAME** downstream pipeline:

```text
Person1 Input (.jsonl file OR Live Camera Stream)
       ↓
Person1Adapter (src/input_adapter.py)
       ↓
NormalizedFrame (src/perception_types.py)
       ↓
TemporalHistoryBuffer (src/temporal_logic.py)
       ↓
ActionInferenceEngine (src/action_inference.py)
       ↓
ObservedAction (State-independent action hypothesis)
       ↓
ExperimentStateMachine (src/state_machine.py) + ProceduralErrorDetector (src/error_detector.py)
       ↓
StateUpdate (Step progression, alerts, errors, completion)
       ├──→ EventLogger (src/event_logger.py)  ──→ [logs/experiment_log_YYYYMMDD_HHMMSS.jsonl]
       ├──→ VoiceAlertManager (src/voice_alert.py) ──→ [Offline Text-To-Speech Output]
       └──→ Monitoring GUI (src/gui.py) ──→ [Real-time Tkinter Interface]
```

---

## 16-Step Experiment Sequence (Source of Truth)

The procedure is stored in [`config/experiment_sequence.json`](file:///c:/Users/SHRAVYA%20HEGDE/OneDrive/Desktop/SIH%20PS%20BAS/config/experiment_sequence.json):

1. `S01 OPEN_WHITE_BOX`: Open the white box.
2. `S02 RETRIEVE_RED_BOX`: Retrieve the red box.
3. `S03 RETRIEVE_YELLOW_BOX`: Retrieve the yellow box.
4. `S04 OPEN_RED_BOX`: Open the red box.
5. `S05 PLANT_TO_WORKPLACE`: Move the plant to the workplace.
6. `S06 OPEN_YELLOW_BOX`: Open the yellow box.
7. `S07 SPRAY_TO_WORKPLACE`: Place the spray bottle at the workplace.
8. `S08 PICK_SPRAY`: Pick up the spray bottle.
9. `S09 SPRAY_PLANT`: Spray the plant.
10. `S10 SPRAY_TO_WORKPLACE_AGAIN`: Return the spray bottle to the workplace.
11. `S11 PLANT_TO_RED_BOX`: Move the plant to the red box.
12. `S12 SPRAY_TO_YELLOW_BOX`: Spray toward the yellow box.
13. `S13 RED_BOX_TO_WHITE_BOX`: Move the red box into the white box.
14. `S14 YELLOW_BOX_TO_WHITE_BOX`: Move the yellow box into the white box.
15. `S15 CLOSE_WHITE_BOX`: Close the white box.
16. `S16 COMPLETE`: Experiment complete.

---

## How to Run

### 1. Run Automated Test Suite (37 Unit Tests)
```bash
python -m unittest discover tests
```

### 2. Launch Monitoring GUI (Interactive Demo)
```bash
python scripts/run_gui_demo.py
```

### 3. Run Full Replay Pipeline on `data/person1_visual_output.jsonl` (1,860 Frames)
```bash
python scripts/run_full_replay.py data/person1_visual_output.jsonl
```

---

## GUI Controls & Features

- **Replay File Selector**: Browse and choose any `.jsonl` perception dataset.
- **Playback Controls**: `START/PLAY`, `PAUSE`, `RESUME`, `STOP`, `RESET`.
- **Replay Speed Multiplier**: `0.25x`, `0.5x`, `1.0x`, `2.0x`, `4.0x`.
- **15-Step Visual Tracker**: Real-time step list highlighting active state ($S01$-$S15$).
- **Structured Event History**: Live treeview showing debounced events (`STEP_COMPLETED`, `SKIPPED_STEP`, `RECOVERY`, etc.).
- **Perception Video Feed**: Display Canvas accepting linked experiment video MP4s (or fallback `"No video source"` label).

---

## Phase 10 — Live Camera Integration & Real-Time Performance

### Architecture & Modes
The system supports three execution input modes:
1. **REPLAY**: Streaming existing JSONL visual perception records.
2. **LIVE_CAMERA**: Direct live video capture via OpenCV (`cv2.VideoCapture`).
3. **MOCK**: Simulated live frame generator for testing without physical camera hardware.

All three modes converge into the exact same `ExperimentPipeline` downstream processing logic.

### Hardware & Setup Requirements
- OpenCV (`opencv-python`)
- Any USB/Webcam video capture device (Default device index `0`)
- 100% Offline execution — no internet or cloud dependencies required.

### Running Live Mode
1. **Launch Monitoring GUI**:
   ```bash
   python scripts/run_gui_demo.py
   ```
2. **Select Input Mode**: Choose `LIVE_CAMERA` or `MOCK` in the top header dropdown.
3. **Camera Controls**:
   - **Cam Device**: Select camera index (0, 1, or 2).
   - **Start Camera**: Starts background capture thread. Display shows live latency and camera status.
   - **Stop Camera**: Releases camera device hardware cleanly.
4. **Experiment Controls**:
   - **START EXPERIMENT**: Initializes state machine ($S01$) and logs `EXPERIMENT_STARTED`.
   - **PAUSE / RESUME**: Temporarily freezes or unfreezes state tracking.
   - **RESET**: Clears state machine, temporal history, metrics, and event log table.

### Live Camera Smoke Test
Run the stand-alone live camera smoke test script:
```bash
python scripts/run_live_camera.py
```

### Live Camera Capture vs. Live Person 1 Perception
- **LIVE CAMERA CAPTURE**: Asynchronous OpenCV video stream (`LiveCameraManager`), thread safety, and real-time FPS/latency telemetry are fully operational.
- **LIVE PERSON 1 PERCEPTION**: Person 1's actual trained detector (`models/best.pt`) and pose estimator (`models/yolo26n-pose.pt`) are fully connected in `src/person1_live_adapter.py`. Real inference runs live per camera frame, producing 6-class object bboxes (`person`, `white_container`, `red_box`, `yellow_box`, `plant`, `spray_bottle`), wrist keypoints (`left_wrist`, `right_wrist`), and hand-object interaction signals.

---

## Phase 10C — Actual Person 1 Model Integration Status

- **Detector Model**: Loaded `models/best.pt` (`task=detect`). Model classes: `{0: 'person', 1: 'white_container', 2: 'red_box', 3: 'yellow_box', 4: 'plant', 5: 'spray_bottle'}`.
- **Pose Model**: Loaded `models/yolo26n-pose.pt` (`task=pose`). COCO 17-keypoint layout (`left_wrist` index 9, `right_wrist` index 10).
- **Status Indicator**: GUI and adapter report `PERSON 1 AI: CONNECTED`.

---

## Known Limitations & Real-World Observations

1. **Replay Dataset Observations (`person1_visual_output.jsonl`)**:
   - Out of 1,860 frames, **1,854 frames (99.7%)** yielded clear action hypotheses, with only **6 frames (0.3%)** marked `UNCERTAIN`.
   - The sequence progresses through $S01 \to S02 \to S04 \to S05 \to S13 \to S14 \to S15$.
   - Slight action sequence skips present in raw dataset video (e.g. spray steps $S07$-$S10$ skipped by operator in this trial) are correctly flagged as `SKIPPED_STEP` errors and safely resynchronized by the state machine.
2. **Live Camera Limitations**:
   - If a physical webcam is not attached to device index 0, the system automatically falls back to `MOCK CAMERA` mode without crashing.
   - Real-time video frame rendering inside Tkinter canvas currently displays camera status text and live metrics banner; raw image bounding box overlays can be directly hooked once Person 1 YOLO bounding box tensors are passed.

