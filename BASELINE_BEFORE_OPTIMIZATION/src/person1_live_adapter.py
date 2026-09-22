"""
Person 1 Live Perception Adapter Interface (Phase 10C)
Integrates trained Person 1 detector (models/best.pt) and pose model (models/yolo26n-pose.pt).
Translates live camera images into Person 1 JSON-compatible output records matching config/PERSON1_OUTPUT_SPEC.txt.
"""
import os
import json
import time
from typing import Dict, Any, Optional, List, Tuple


class Person1LiveAdapter:
    """
    Adapter boundary between Person 1 CV detection & pose models and Person 2 state-machine pipeline.
    Loads models/best.pt and models/yolo26n-pose.pt using ultralytics YOLO.
    """

    EXPECTED_CLASSES = [
        "person",
        "white_container",
        "red_box",
        "yellow_box",
        "plant",
        "spray_bottle"
    ]

    def __init__(
        self,
        config_path: str = "config/person1_live.json",
        mock_mode: bool = False,
        custom_model: Optional[Any] = None,
        custom_pose_model: Optional[Any] = None
    ):
        self.config_path = config_path
        self.mock_mode = mock_mode
        self.is_cv_model_connected: bool = False
        self.model = custom_model
        self.pose_model = custom_pose_model
        self.adapter_status: str = "INITIALIZING"
        self.load_error_message: Optional[str] = None
        self.model_classes: Dict[int, str] = {}

        # Default configuration
        self.config: Dict[str, Any] = {
            "model_path": "models/best.pt",
            "pose_model_path": "models/yolo26n-pose.pt",
            "device": "cpu",
            "confidence_threshold": 0.35,
            "image_size": 640,
            "tracking_enabled": True,
            "pose_skip_cadence": 1
        }

        self._last_pose_dict: Dict[str, Any] = {}

        self._load_config()

        if self.model is not None:
            self.is_cv_model_connected = True
            self.adapter_status = "CONNECTED"
            if hasattr(self.model, 'names'):
                self.model_classes = self.model.names
        elif not self.mock_mode:
            self._try_load_models()

    def _load_config(self) -> None:
        """Loads configuration from JSON file if available."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    user_cfg = json.load(f)
                    self.config.update(user_cfg)
            except Exception as e:
                self.load_error_message = f"Config load warning: {e}"

    def _try_load_models(self) -> None:
        """
        Loads local Person 1 trained detector (models/best.pt) and pose model (models/yolo26n-pose.pt).
        """
        model_path = self.config.get("model_path", "models/best.pt")
        pose_path = self.config.get("pose_model_path", "models/yolo26n-pose.pt")

        if not os.path.exists(model_path):
            self.is_cv_model_connected = False
            self.adapter_status = "PERCEPTION_NOT_CONNECTED"
            self.load_error_message = f"Detector model file not found: '{model_path}'"
            return

        try:
            from ultralytics import YOLO

            # 1. Load Object Detector Model
            self.model = YOLO(model_path)
            if hasattr(self.model, 'names'):
                self.model_classes = self.model.names
                print(f"[Person1LiveAdapter] Loaded detector '{model_path}'. Model classes: {self.model_classes}")

            # 2. Load Pose Model if present
            if os.path.exists(pose_path):
                try:
                    self.pose_model = YOLO(pose_path)
                    print(f"[Person1LiveAdapter] Loaded pose model '{pose_path}'.")
                except Exception as pe:
                    self.pose_model = None
                    print(f"[Person1LiveAdapter] Warning loading pose model '{pose_path}': {pe}")
            else:
                self.pose_model = None

            self.is_cv_model_connected = True
            self.adapter_status = "CONNECTED"
            self.load_error_message = None

        except Exception as e:
            self.is_cv_model_connected = False
            self.adapter_status = "PERCEPTION_NOT_CONNECTED"
            self.load_error_message = str(e)
            print(f"[Person1LiveAdapter] Failed to initialize YOLO models: {e}")

    def process_live_frame(
        self,
        frame_num: int,
        timestamp: float,
        image_matrix: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Runs perception model on live image frame and returns PERSON1_OUTPUT_SPEC compatible dictionary.
        """
        if self.mock_mode:
            return self._generate_mock_perception_record(frame_num, timestamp)

        if not self.is_cv_model_connected or self.model is None or image_matrix is None:
            return {
                "frame": frame_num,
                "timestamp": timestamp,
                "objects": [],
                "pose": {},
                "hand_object_geometry": {},
                "hand_object_interaction": {},
                "interaction_signals": {},
                "adapter_status": "PERCEPTION_NOT_CONNECTED",
                "error_detail": self.load_error_message
            }

        try:
            return self._run_inference(frame_num, timestamp, image_matrix)
        except Exception as e:
            return {
                "frame": frame_num,
                "timestamp": timestamp,
                "objects": [],
                "pose": {},
                "hand_object_geometry": {},
                "hand_object_interaction": {},
                "interaction_signals": {},
                "adapter_status": "INFERENCE_ERROR",
                "error_detail": str(e)
            }

    def _run_inference(self, frame_num: int, timestamp: float, image_matrix: Any) -> Dict[str, Any]:
        """Executes actual detector & pose inference on image_matrix."""
        conf = self.config.get("confidence_threshold", 0.35)
        imgsz = self.config.get("image_size", 640)

        # 1. Run Object Detector / Tracker
        objects = []
        parsed_boxes = {}

        if self.config.get("tracking_enabled", True) and hasattr(self.model, 'track'):
            try:
                results = self.model.track(image_matrix, conf=conf, imgsz=imgsz, persist=True, verbose=False)
            except Exception:
                results = self.model(image_matrix, conf=conf, imgsz=imgsz, verbose=False)
        else:
            results = self.model(image_matrix, conf=conf, imgsz=imgsz, verbose=False)

        if results and len(results) > 0:
            res = results[0]
            boxes = res.boxes
            if boxes is not None and len(boxes) > 0:
                for idx, b in enumerate(boxes):
                    cls_id = int(b.cls[0].item()) if hasattr(b.cls, '__getitem__') else int(b.cls.item())
                    cls_name = self.model.names.get(cls_id, f"class_{cls_id}")
                    confidence = float(b.conf[0].item()) if hasattr(b.conf, '__getitem__') else float(b.conf.item())
                    track_id = int(b.id[0].item()) if (b.id is not None) else (idx + 1)
                    xyxy = b.xyxy[0].tolist() if hasattr(b.xyxy[0], 'tolist') else list(b.xyxy[0])

                    obj_item = {
                        "class": cls_name,
                        "track_id": track_id,
                        "confidence": round(confidence, 2),
                        "bbox": [round(v, 1) for v in xyxy]
                    }
                    objects.append(obj_item)
                    parsed_boxes[cls_name] = obj_item

        # 2. Run Pose Estimator (with configurable cadence optimization)
        cadence = max(1, int(self.config.get("pose_skip_cadence", 1)))
        pose_dict = self._last_pose_dict

        if self.pose_model is not None and (frame_num % cadence == 0 or not self._last_pose_dict):
            try:
                pose_res = self.pose_model(image_matrix, conf=conf, imgsz=imgsz, verbose=False)
                if pose_res and len(pose_res) > 0 and hasattr(pose_res[0], 'keypoints') and pose_res[0].keypoints is not None:
                    kpts = pose_res[0].keypoints.data
                    if kpts is not None and len(kpts) > 0:
                        person_kpts = kpts[0]
                        if hasattr(person_kpts, 'tolist'):
                            person_kpts = person_kpts.tolist()
                        if len(person_kpts) >= 11:  # Index 9 = left wrist, Index 10 = right wrist
                            lw_data = person_kpts[9]
                            rw_data = person_kpts[10]

                            def _val(item):
                                return float(item.item() if hasattr(item, 'item') else item)

                            pose_dict = {
                                "person_track_id": 1,
                                "left_wrist": {
                                    "x": round(_val(lw_data[0]), 1),
                                    "y": round(_val(lw_data[1]), 1),
                                    "confidence": round(_val(lw_data[2]), 2)
                                },
                                "right_wrist": {
                                    "x": round(_val(rw_data[0]), 1),
                                    "y": round(_val(rw_data[1]), 1),
                                    "confidence": round(_val(rw_data[2]), 2)
                                }
                            }
                            self._last_pose_dict = pose_dict
            except Exception as pe:
                pose_dict = self._last_pose_dict

        # 3. Compute Hand-Object Geometry & Proximity Signals
        interaction_signals = {}
        hand_object_interaction = {"left": {}, "right": {}}

        rw = pose_dict.get("right_wrist")
        lw = pose_dict.get("left_wrist")

        for cls_name in ["white_container", "red_box", "yellow_box", "plant", "spray_bottle"]:
            obj = parsed_boxes.get(cls_name)
            if obj and obj.get("bbox"):
                bbox = obj["bbox"]
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0

                if rw and rw.get("confidence", 0) > 0.25:
                    dist_r = ((rw["x"] - cx)**2 + (rw["y"] - cy)**2)**0.5
                    near_r = dist_r < 150.0
                    if near_r:
                        interaction_signals[f"right_hand_near_{cls_name}"] = True
                        hand_object_interaction["right"][cls_name] = {
                            "near": True,
                            "stable": True,
                            "track_id": obj["track_id"],
                            "distance_px": round(dist_r, 1)
                        }

                if lw and lw.get("confidence", 0) > 0.25:
                    dist_l = ((lw["x"] - cx)**2 + (lw["y"] - cy)**2)**0.5
                    near_l = dist_l < 150.0
                    if near_l:
                        interaction_signals[f"left_hand_near_{cls_name}"] = True
                        hand_object_interaction["left"][cls_name] = {
                            "near": True,
                            "stable": True,
                            "track_id": obj["track_id"],
                            "distance_px": round(dist_l, 1)
                        }

        return {
            "frame": frame_num,
            "timestamp": timestamp,
            "objects": objects,
            "pose": pose_dict,
            "hand_object_geometry": {},
            "hand_object_interaction": hand_object_interaction,
            "interaction_signals": interaction_signals,
            "adapter_status": "CONNECTED"
        }

    def _generate_mock_perception_record(self, frame_num: int, timestamp: float) -> Dict[str, Any]:
        """Generates valid mock perception record for development/testing when explicit MOCK mode is chosen."""
        return {
            "frame": frame_num,
            "timestamp": timestamp,
            "objects": [
                {"class": "person", "track_id": 1, "confidence": 0.92, "bbox": [100.0, 100.0, 500.0, 900.0]},
                {"class": "white_container", "track_id": 2, "confidence": 0.88, "bbox": [200.0, 600.0, 600.0, 900.0]}
            ],
            "pose": {
                "person_track_id": 1,
                "left_wrist": {"x": 400.0, "y": 700.0, "confidence": 0.85},
                "right_wrist": {"x": 450.0, "y": 720.0, "confidence": 0.88}
            },
            "hand_object_geometry": {},
            "hand_object_interaction": {
                "right": {
                    "white_container": {"near": True, "stable": True, "track_id": 2, "distance_px": 15.0}
                }
            },
            "interaction_signals": {
                "right_hand_near_white_container": True
            },
            "adapter_status": "MOCK_ACTIVE"
        }
