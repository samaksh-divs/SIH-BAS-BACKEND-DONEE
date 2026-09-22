"""
Performance Profiling and Optimization Audit Script (Part B & Part C).
Profiles execution breakdown of:
- Detector inference
- Pose inference
- Tracking / bounding box parsing
- Interaction calculations
- Person 2 state pipeline processing
Compares BEFORE (every frame pose) vs AFTER (pose skip cadence).
"""
import os
import sys
import time
import json
import numpy as np

sys.path.insert(0, os.path.abspath('.'))
from src.person1_live_adapter import Person1LiveAdapter
from src.pipeline import ExperimentPipeline

def profile_run(adapter: Person1LiveAdapter, pipeline: ExperimentPipeline, frames_count: int = 50, pose_cadence: int = 1):
    # Dummy frame matrix (640x480)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    det_times = []
    pose_times = []
    spatial_times = []
    p2_times = []
    total_times = []

    last_pose_dict = {}

    for f in range(1, frames_count + 1):
        t_start = time.time()
        
        # 1. Detector inference
        t0 = time.time()
        conf = adapter.config.get("confidence_threshold", 0.35)
        imgsz = adapter.config.get("image_size", 640)
        
        results = adapter.model(dummy_frame, conf=conf, imgsz=imgsz, verbose=False)
        parsed_boxes = {}
        objects = []
        if results and len(results) > 0:
            res = results[0]
            boxes = res.boxes
            if boxes is not None and len(boxes) > 0:
                for idx, b in enumerate(boxes):
                    cls_id = int(b.cls[0].item())
                    cls_name = adapter.model.names.get(cls_id, f"class_{cls_id}")
                    confidence = float(b.conf[0].item())
                    xyxy = b.xyxy[0].tolist()
                    obj_item = {
                        "class": cls_name,
                        "track_id": idx + 1,
                        "confidence": round(confidence, 2),
                        "bbox": [round(v, 1) for v in xyxy]
                    }
                    objects.append(obj_item)
                    parsed_boxes[cls_name] = obj_item
        t1 = time.time()
        det_times.append((t1 - t0) * 1000.0)

        # 2. Pose inference (with skip cadence)
        t2_start = time.time()
        if adapter.pose_model is not None and (f % pose_cadence == 0 or not last_pose_dict):
            try:
                pose_res = adapter.pose_model(dummy_frame, conf=conf, imgsz=imgsz, verbose=False)
                if pose_res and len(pose_res) > 0 and hasattr(pose_res[0], 'keypoints') and pose_res[0].keypoints is not None:
                    kpts = pose_res[0].keypoints.data
                    if kpts is not None and len(kpts) > 0:
                        person_kpts = kpts[0]
                        if hasattr(person_kpts, 'tolist'):
                            person_kpts = person_kpts.tolist()
                        if len(person_kpts) >= 11:
                            lw_data = person_kpts[9]
                            rw_data = person_kpts[10]
                            def _val(v): return float(v.item() if hasattr(v, 'item') else v)
                            last_pose_dict = {
                                "person_track_id": 1,
                                "left_wrist": {"x": round(_val(lw_data[0]), 1), "y": round(_val(lw_data[1]), 1), "confidence": round(_val(lw_data[2]), 2)},
                                "right_wrist": {"x": round(_val(rw_data[0]), 1), "y": round(_val(rw_data[1]), 1), "confidence": round(_val(rw_data[2]), 2)}
                            }
            except Exception:
                pass
        t2_end = time.time()
        pose_times.append((t2_end - t2_start) * 1000.0)

        # 3. Spatial interaction calculations
        t3_start = time.time()
        interaction_signals = {}
        hand_object_interaction = {"left": {}, "right": {}}
        rw = last_pose_dict.get("right_wrist")
        lw = last_pose_dict.get("left_wrist")
        for cls_name in ["white_container", "red_box", "yellow_box", "plant", "spray_bottle"]:
            obj = parsed_boxes.get(cls_name)
            if obj and obj.get("bbox"):
                bbox = obj["bbox"]
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                if rw and rw.get("confidence", 0) > 0.25:
                    dist_r = ((rw["x"] - cx)**2 + (rw["y"] - cy)**2)**0.5
                    if dist_r < 150.0:
                        interaction_signals[f"right_hand_near_{cls_name}"] = True
                        hand_object_interaction["right"][cls_name] = {"near": True, "stable": True, "track_id": obj["track_id"], "distance_px": round(dist_r, 1)}
                if lw and lw.get("confidence", 0) > 0.25:
                    dist_l = ((lw["x"] - cx)**2 + (lw["y"] - cy)**2)**0.5
                    if dist_l < 150.0:
                        interaction_signals[f"left_hand_near_{cls_name}"] = True
                        hand_object_interaction["left"][cls_name] = {"near": True, "stable": True, "track_id": obj["track_id"], "distance_px": round(dist_l, 1)}
        t3_end = time.time()
        spatial_times.append((t3_end - t3_start) * 1000.0)

        perc_record = {
            "frame": f,
            "timestamp": f * 0.033,
            "objects": objects,
            "pose": last_pose_dict,
            "hand_object_geometry": {},
            "hand_object_interaction": hand_object_interaction,
            "interaction_signals": interaction_signals,
            "adapter_status": "CONNECTED"
        }

        # 4. Person 2 processing
        t4_start = time.time()
        pipeline.process_frame(perc_record)
        t4_end = time.time()
        p2_times.append((t4_end - t4_start) * 1000.0)

        t_end = time.time()
        total_times.append((t_end - t_start) * 1000.0)

    # Calculate statistics
    avg_det = np.mean(det_times)
    avg_pose = np.mean(pose_times)
    avg_spatial = np.mean(spatial_times)
    avg_p2 = np.mean(p2_times)
    avg_total = np.mean(total_times)
    p50_total = np.percentile(total_times, 50)
    p95_total = np.percentile(total_times, 95)
    max_total = np.max(total_times)
    fps = 1000.0 / avg_total if avg_total > 0 else 0.0

    return {
        "pose_cadence": pose_cadence,
        "avg_detector_ms": round(float(avg_det), 2),
        "avg_pose_ms": round(float(avg_pose), 2),
        "avg_spatial_ms": round(float(avg_spatial), 2),
        "avg_p2_ms": round(float(avg_p2), 2),
        "avg_total_ms": round(float(avg_total), 2),
        "p50_total_ms": round(float(p50_total), 2),
        "p95_total_ms": round(float(p95_total), 2),
        "max_total_ms": round(float(max_total), 2),
        "end_to_end_fps": round(float(fps), 2)
    }

def main():
    print("========================================")
    print("PART B & C — PERFORMANCE OPTIMIZATION AUDIT")
    print("========================================")
    
    adapter = Person1LiveAdapter(config_path="config/person1_live.json")
    if not adapter.is_cv_model_connected:
        print(f"Error loading models: {adapter.load_error_message}")
        return

    pipeline = ExperimentPipeline()

    print("\nWarmup (5 frames)...")
    profile_run(adapter, pipeline, frames_count=5, pose_cadence=1)

    print("\nProfiling BASELINE (Pose Every Frame, cadence=1)...")
    base_res = profile_run(adapter, pipeline, frames_count=40, pose_cadence=1)
    
    print("\nProfiling OPTIMIZED (Pose Every 2nd Frame, cadence=2)...")
    opt_res = profile_run(adapter, pipeline, frames_count=40, pose_cadence=2)

    print("\n----------------------------------------")
    print("PROFILING BREAKDOWN SUMMARY:")
    print("----------------------------------------")
    print(f"BEFORE (Baseline Cadence = 1):")
    print(f"  Detector Avg:   {base_res['avg_detector_ms']} ms")
    print(f"  Pose Avg:       {base_res['avg_pose_ms']} ms")
    print(f"  Spatial Calc:   {base_res['avg_spatial_ms']} ms")
    print(f"  Person 2 Logic: {base_res['avg_p2_ms']} ms")
    print(f"  Total Latency:  {base_res['avg_total_ms']} ms (P50: {base_res['p50_total_ms']} ms, P95: {base_res['p95_total_ms']} ms, Max: {base_res['max_total_ms']} ms)")
    print(f"  End-to-End FPS: {base_res['end_to_end_fps']} FPS")
    
    print(f"\nAFTER (Optimized Cadence = 2):")
    print(f"  Detector Avg:   {opt_res['avg_detector_ms']} ms")
    print(f"  Pose Avg:       {opt_res['avg_pose_ms']} ms")
    print(f"  Spatial Calc:   {opt_res['avg_spatial_ms']} ms")
    print(f"  Person 2 Logic: {opt_res['avg_p2_ms']} ms")
    print(f"  Total Latency:  {opt_res['avg_total_ms']} ms (P50: {opt_res['p50_total_ms']} ms, P95: {opt_res['p95_total_ms']} ms, Max: {opt_res['max_total_ms']} ms)")
    print(f"  End-to-End FPS: {opt_res['end_to_end_fps']} FPS")

    # Save results json
    report = {
        "baseline": base_res,
        "optimized": opt_res
    }
    os.makedirs("logs", exist_ok=True)
    with open("logs/performance_audit.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("\nSaved breakdown to logs/performance_audit.json")

if __name__ == "__main__":
    main()
