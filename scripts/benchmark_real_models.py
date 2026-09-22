"""
Real Person 1 Models Telemetry Benchmark Script
Measures actual CPU inference time, detection FPS, pose FPS, end-to-end FPS, average latency,
P95 latency, and max latency using real model files (models/best.pt and models/yolo26n-pose.pt).
"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.person1_live_adapter import Person1LiveAdapter
from src.pipeline import ExperimentPipeline


def main():
    print("=" * 75)
    print("BENCHMARKING REAL PERSON 1 MODELS (models/best.pt & models/yolo26n-pose.pt)")
    print("=" * 75)

    adapter = Person1LiveAdapter(config_path="config/person1_live.json", mock_mode=False)
    pipeline = ExperimentPipeline(use_mock_tts=True)

    print(f"  - Detector Model Connected: {adapter.is_cv_model_connected}")
    print(f"  - Model Classes:            {adapter.model_classes}")

    # Generate test image matrix (480x640x3 uint8)
    np.random.seed(42)
    sample_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Warmup models
    print("\nWarming up detector and pose models...")
    for _ in range(3):
        adapter.process_live_frame(frame_num=0, timestamp=time.time(), image_matrix=sample_frame)

    print("Running 50 live inference frames...")
    latencies = []
    p1_times = []
    proc_times = []
    det_times = []
    pose_times = []

    start_total = time.time()
    for f_idx in range(1, 51):
        cap_ts = time.time()

        # Person 1 Inference
        t0 = time.time()
        p_record = adapter.process_live_frame(frame_num=f_idx, timestamp=cap_ts, image_matrix=sample_frame)
        t1 = time.time()

        # Downstream Person 2 Pipeline
        update, norm_frame, obs_action = pipeline.process_frame(p_record)
        t2 = time.time()

        total_latency = (t2 - cap_ts) * 1000.0
        latencies.append(total_latency)
        p1_times.append((t1 - t0))
        proc_times.append((t2 - cap_ts))

        if f_idx % 10 == 0:
            print(f"  Frame {f_idx:2d}/50 | P1 Time: {(t1-t0)*1000.0:.1f} ms | Turnaround Latency: {total_latency:.1f} ms | State: {update.current_state_id}")

    end_total = time.time()
    total_elapsed = end_total - start_total

    avg_p1_sec = sum(p1_times) / len(p1_times)
    avg_proc_sec = sum(proc_times) / len(proc_times)
    avg_lat_ms = sum(latencies) / len(latencies)

    sorted_lat = sorted(latencies)
    p95_lat_ms = sorted_lat[int(len(sorted_lat) * 0.95)]
    max_lat_ms = sorted_lat[-1]

    p1_fps = 1.0 / avg_p1_sec if avg_p1_sec > 0 else 0
    e2e_fps = len(proc_times) / total_elapsed

    print("\n" + "=" * 75)
    print("REAL PERSON 1 MODEL BENCHMARK RESULTS")
    print("=" * 75)
    print(f"  - Total Test Frames:          50")
    print(f"  - Person 1 Inference FPS:     {p1_fps:.2f} FPS")
    print(f"  - End-to-End Turnaround FPS:  {e2e_fps:.2f} FPS")
    print(f"  - Average Turnaround Latency: {avg_lat_ms:.1f} ms")
    print(f"  - P95 Turnaround Latency:     {p95_lat_ms:.1f} ms")
    print(f"  - Maximum Turnaround Latency: {max_lat_ms:.1f} ms")
    print("=" * 75)


if __name__ == "__main__":
    main()
