"""
Person 1 Live Perception Adapter Interface (Phase 10C)

Integrates:
- Person 1 trained detector: models/best.pt
- Person 1 zero-shot detector: models/yolov8s-world.pt (YOLO-World)
- Pose model: models/yolo26n-pose.pt

Translates live camera images into Person 1 JSON-compatible
output records matching config/PERSON1_OUTPUT_SPEC.txt.
"""

import os
import json
from typing import Dict, Any, Optional


class Person1LiveAdapter:
    """
    Adapter boundary between Person 1 CV detection/pose models
    and the Person 2 state-machine pipeline.
    """

    EXPECTED_CLASSES = [
        "person",
        "white_container",
        "red_box",
        "yellow_box",
        "plant",
        "spray_bottle"
    ]

    # YOLO-World zero-shot prompt classes (exact model-side names)
    YOLO_WORLD_CLASSES = [
        "person",
        "white rectangular box",
        "yellow box",
        "red box",
        "plant",
        "hand spray bottle"
    ]

    # Backward-compatible naming convention expected by the
    # downstream BAS pipeline (spaces -> underscores).
    CLASS_NAME_NORMALIZATION = {
        "white rectangular box": "white_container",
        "white container": "white_container",
        "red box": "red_box",
        "yellow box": "yellow_box",
        "black spray bottle": "spray_bottle",
        "spray bottle": "spray_bottle",
        "hand spray bottle": "spray_bottle"
    }

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

        # Resolved inference device ("cpu" unless config + CUDA agree)
        self.device: str = "cpu"

        # True only when models were loaded internally by this adapter;
        # externally supplied models (tests / custom) keep their own
        # call signatures, so device is not forwarded to them.
        self._models_loaded_internally: bool = False

        self._load_config()

        # Custom model supplied by tests or external caller
        if self.model is not None:
            self.is_cv_model_connected = True
            self.adapter_status = "CONNECTED"

            if hasattr(self.model, "names"):
                self.model_classes = self.model.names

        # Otherwise load actual models
        elif not self.mock_mode:
            self._try_load_models()

        # Explicit mock mode
        else:
            self.adapter_status = "MOCK_ACTIVE"

    def _load_config(self) -> None:
        """Loads configuration from JSON file if available."""

        if os.path.exists(self.config_path):
            try:
                with open(
                    self.config_path,
                    "r",
                    encoding="utf-8"
                ) as f:
                    user_cfg = json.load(f)

                self.config.update(user_cfg)

            except Exception as e:
                self.load_error_message = (
                    f"Config load warning: {e}"
                )

    def _try_load_models(self) -> None:
        """
        Loads the local Person 1 detector and pose model.

        Detector:
            models/best.pt

        Pose:
            models/yolo26n-pose.pt
        """

        model_path = self.config.get(
            "model_path",
            "models/best.pt"
        )

        pose_path = self.config.get(
            "pose_model_path",
            "models/yolo26n-pose.pt"
        )

        try:
            # ---------------------------------------------------------
            # 1. Check detector model
            # ---------------------------------------------------------
            # Resolve device from config with a safe fallback:
            # never blindly force CUDA.
            requested_device = self.config.get("device", "cpu")
            try:
                import torch

                device = (
                    requested_device
                    if (
                        requested_device == "cpu"
                        or torch.cuda.is_available()
                    )
                    else "cpu"
                )

            except Exception:
                device = "cpu"

            self.device = device

            if not os.path.exists(model_path):
                self.is_cv_model_connected = False
                self.adapter_status = "PERCEPTION_NOT_CONNECTED"

                self.load_error_message = (
                    f"Detector model file not found: '{model_path}'"
                )

                print(
                    f"[Person1LiveAdapter] "
                    f"{self.load_error_message}"
                )

                return

            # ---------------------------------------------------------
            # 2. Import Ultralytics
            # ---------------------------------------------------------
            from ultralytics import YOLO

            # ---------------------------------------------------------
            # 3. Load object detector
            # ---------------------------------------------------------
            if (
                "world" in model_path.lower()
                or "world" in self.config_path.lower()
            ):
                from ultralytics import YOLOWorld

                self.model = YOLOWorld(model_path)

                # YOLO-World zero-shot classes
                self.model.set_classes(self.YOLO_WORLD_CLASSES)

                if hasattr(self.model, "names"):
                    self.model_classes = self.model.names

                # Keep YOLO-World weights on the configured device so
                # the CLIP text-prompt embeddings stay in sync with
                # inference inputs (no cross-device copies per frame).
                try:
                    if hasattr(self.model, "to"):
                        self.model.to(device)
                except Exception as de:
                    print(
                        f"[Person1LiveAdapter] "
                        f"Device move warning for "
                        f"'{model_path}': {de}"
                    )

                print(
                    f"[Person1LiveAdapter] "
                    f"Loaded YOLO-World '{model_path}'. "
                    f"Zero-shot classes: {self.model_classes}"
                )

            else:
                self.model = YOLO(model_path)

                if hasattr(self.model, "names"):
                    self.model_classes = self.model.names

                print(
                    f"[Person1LiveAdapter] "
                    f"Loaded detector '{model_path}'. "
                    f"Model classes: {self.model_classes}"
                )

            # ---------------------------------------------------------
            # 4. Load pose model
            # ---------------------------------------------------------
            if os.path.exists(pose_path):
                try:
                    self.pose_model = YOLO(pose_path)

                    print(
                        f"[Person1LiveAdapter] "
                        f"Loaded pose model '{pose_path}'."
                    )

                except Exception as pe:
                    self.pose_model = None

                    print(
                        f"[Person1LiveAdapter] "
                        f"Warning loading pose model "
                        f"'{pose_path}': {pe}"
                    )

            else:
                self.pose_model = None

                print(
                    f"[Person1LiveAdapter] "
                    f"Pose model not found: '{pose_path}'"
                )

            # ---------------------------------------------------------
            # 5. Mark adapter connected
            # ---------------------------------------------------------
            self._models_loaded_internally = True
            self.is_cv_model_connected = True
            self.adapter_status = "CONNECTED"
            self.load_error_message = None

            print(
                "[Person1LiveAdapter] "
                "PERSON 1 AI: CONNECTED"
            )

        except Exception as e:
            self.is_cv_model_connected = False
            self.adapter_status = "PERCEPTION_NOT_CONNECTED"
            self.load_error_message = str(e)

            print(
                "[Person1LiveAdapter] "
                f"Failed to initialize YOLO models: {e}"
            )

    def process_live_frame(
        self,
        frame_num: int,
        timestamp: float,
        image_matrix: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Runs perception model on a live image frame and returns
        a PERSON1_OUTPUT_SPEC-compatible dictionary.
        """

        # -------------------------------------------------------------
        # MOCK MODE
        # -------------------------------------------------------------
        if self.mock_mode:
            return self._generate_mock_perception_record(
                frame_num,
                timestamp
            )

        # -------------------------------------------------------------
        # Model unavailable
        # -------------------------------------------------------------
        if (
            not self.is_cv_model_connected
            or self.model is None
            or image_matrix is None
        ):
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

        # -------------------------------------------------------------
        # Actual inference
        # -------------------------------------------------------------
        try:
            return self._run_inference(
                frame_num,
                timestamp,
                image_matrix
            )

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

    def _run_inference(
        self,
        frame_num: int,
        timestamp: float,
        image_matrix: Any
    ) -> Dict[str, Any]:
        """
        Executes actual detector and pose inference.
        """

        conf = self.config.get(
            "confidence_threshold",
            0.35
        )

        imgsz = self.config.get(
            "image_size",
            640
        )

        # =============================================================
        # 1. OBJECT DETECTION / TRACKING
        # =============================================================

        objects = []
        parsed_boxes = {}

        # Only forward the configured device to internally-loaded
        # (real Ultralytics) models. Externally supplied models such
        # as test mocks keep their original call signature.
        device_kwargs = (
            {"device": self.device}
            if self._models_loaded_internally
            else {}
        )
        if (
            self.config.get("tracking_enabled", True)
            and hasattr(self.model, "track")
        ):
            try:
                results = self.model.track(
                    image_matrix,
                    conf=conf,
                    imgsz=imgsz,
                    persist=True,
                    **device_kwargs,
                    verbose=False
                )

            except Exception:
                results = self.model(
                    image_matrix,
                    conf=conf,
                    imgsz=imgsz,
                    **device_kwargs,
                    verbose=False
                )

        else:
            results = self.model(
                image_matrix,
                conf=conf,
                imgsz=imgsz,
                **device_kwargs,
                verbose=False
            )

        # -------------------------------------------------------------
        # Parse detector output
        # -------------------------------------------------------------
        if results and len(results) > 0:

            res = results[0]
            boxes = getattr(res, "boxes", None)

            if boxes is not None and len(boxes) > 0:

                for idx, b in enumerate(boxes):

                    # Class ID
                    if hasattr(b.cls, "__getitem__"):
                        cls_id = int(
                            b.cls[0].item()
                        )
                    else:
                        cls_id = int(
                            b.cls.item()
                        )

                    # Class name
                    names = getattr(
                        self.model,
                        "names",
                        self.model_classes
                    )

                    if isinstance(names, dict):
                        cls_name = names.get(
                            cls_id,
                            f"class_{cls_id}"
                        )
                    else:
                        try:
                            cls_name = names[cls_id]
                        except Exception:
                            cls_name = f"class_{cls_id}"

                    # Confidence
                    if hasattr(b.conf, "__getitem__"):
                        confidence = float(
                            b.conf[0].item()
                        )
                    else:
                        confidence = float(
                            b.conf.item()
                        )

                    # Track ID
                    if b.id is not None:
                        try:
                            track_id = int(
                                b.id[0].item()
                            )
                        except Exception:
                            track_id = idx + 1
                    else:
                        track_id = idx + 1

                    # Bounding box
                    xyxy = (
                        b.xyxy[0].tolist()
                        if hasattr(
                            b.xyxy[0],
                            "tolist"
                        )
                        else list(b.xyxy[0])
                    )

                    # -------------------------------------------------
                    # Normalize class names
                    # -------------------------------------------------
                    cls_name = self.CLASS_NAME_NORMALIZATION.get(
                        cls_name,
                        cls_name
                    )

                    # -------------------------------------------------
                    # Prototype spray-bottle color reclassification
                    # -------------------------------------------------
                    if (
                        cls_name == "red_box"
                        and image_matrix is not None
                    ):
                        try:
                            import numpy as np

                            x1c, y1c, x2c, y2c = [
                                max(0, int(v))
                                for v in xyxy
                            ]

                            roi = image_matrix[
                                y1c:y2c,
                                x1c:x2c
                            ]

                            if roi.size > 0:

                                mean_bgr = (
                                    roi.reshape(-1, 3)
                                    .mean(axis=0)
                                )

                                brightness = (
                                    mean_bgr.mean()
                                )

                                red_dominance = (
                                    mean_bgr[2]
                                    - max(
                                        mean_bgr[0],
                                        mean_bgr[1]
                                    )
                                )

                                if (
                                    brightness < 80
                                    or red_dominance < 15
                                ):
                                    cls_name = (
                                        "spray_bottle"
                                    )

                        except Exception:
                            pass

                    # -------------------------------------------------
                    # Create normalized object
                    # -------------------------------------------------
                    obj_item = {
                        "class": cls_name,
                        "track_id": track_id,
                        "confidence": round(
                            confidence,
                            2
                        ),
                        "bbox": [
                            round(v, 1)
                            for v in xyxy
                        ]
                    }

                    # IMPORTANT:
                    # Actually add object to output.
                    objects.append(obj_item)

                    # IMPORTANT:
                    # Store latest object of each class for
                    # hand-object interaction calculations.
                    parsed_boxes[cls_name] = obj_item

        # =============================================================
        # 2. POSE ESTIMATION
        # =============================================================

        cadence = max(
            1,
            int(
                self.config.get(
                    "pose_skip_cadence",
                    1
                )
            )
        )

        pose_dict = self._last_pose_dict

        if (
            self.pose_model is not None
            and (
                frame_num % cadence == 0
                or not self._last_pose_dict
            )
        ):

            try:
                pose_res = self.pose_model(
                    image_matrix,
                    conf=conf,
                    imgsz=imgsz,
                    **device_kwargs,
                    verbose=False
                )

                if (
                    pose_res
                    and len(pose_res) > 0
                    and hasattr(
                        pose_res[0],
                        "keypoints"
                    )
                    and pose_res[0].keypoints is not None
                ):

                    kpts = (
                        pose_res[0]
                        .keypoints
                        .data
                    )

                    if (
                        kpts is not None
                        and len(kpts) > 0
                    ):

                        person_kpts = kpts[0]

                        if hasattr(
                            person_kpts,
                            "tolist"
                        ):
                            person_kpts = (
                                person_kpts.tolist()
                            )

                        # COCO:
                        # index 9  = left wrist
                        # index 10 = right wrist
                        if len(person_kpts) >= 11:

                            lw_data = person_kpts[9]
                            rw_data = person_kpts[10]

                            def _val(item):
                                return float(
                                    item.item()
                                    if hasattr(
                                        item,
                                        "item"
                                    )
                                    else item
                                )

                            pose_dict = {
                                "person_track_id": 1,

                                "left_wrist": {
                                    "x": round(
                                        _val(
                                            lw_data[0]
                                        ),
                                        1
                                    ),
                                    "y": round(
                                        _val(
                                            lw_data[1]
                                        ),
                                        1
                                    ),
                                    "confidence": round(
                                        _val(
                                            lw_data[2]
                                        ),
                                        2
                                    )
                                },

                                "right_wrist": {
                                    "x": round(
                                        _val(
                                            rw_data[0]
                                        ),
                                        1
                                    ),
                                    "y": round(
                                        _val(
                                            rw_data[1]
                                        ),
                                        1
                                    ),
                                    "confidence": round(
                                        _val(
                                            rw_data[2]
                                        ),
                                        2
                                    )
                                }
                            }

                            self._last_pose_dict = (
                                pose_dict
                            )

            except Exception as e:
                # Keep last valid pose instead of
                # breaking the entire perception pipeline.
                print(
                    f"[Person1LiveAdapter] "
                    f"Pose inference warning: {e}"
                )

        # =============================================================
        # 3. HAND-OBJECT GEOMETRY / INTERACTION
        # =============================================================

        interaction_signals = {}

        hand_object_interaction = {
            "left": {},
            "right": {}
        }

        rw = pose_dict.get(
            "right_wrist"
        )

        lw = pose_dict.get(
            "left_wrist"
        )

        tracked_classes = [
            "white_container",
            "red_box",
            "yellow_box",
            "plant",
            "spray_bottle"
        ]

        for cls_name in tracked_classes:

            obj = parsed_boxes.get(
                cls_name
            )

            if obj and obj.get("bbox"):

                bbox = obj["bbox"]

                cx = (
                    bbox[0]
                    + bbox[2]
                ) / 2.0

                cy = (
                    bbox[1]
                    + bbox[3]
                ) / 2.0

                # -----------------------------------------------------
                # Right hand
                # -----------------------------------------------------
                if (
                    rw
                    and rw.get(
                        "confidence",
                        0
                    ) > 0.25
                ):

                    dist_r = (
                        (
                            rw["x"] - cx
                        ) ** 2
                        +
                        (
                            rw["y"] - cy
                        ) ** 2
                    ) ** 0.5

                    near_r = (
                        dist_r < 150.0
                    )

                    if near_r:

                        interaction_signals[
                            f"right_hand_near_{cls_name}"
                        ] = True

                        hand_object_interaction[
                            "right"
                        ][cls_name] = {
                            "near": True,
                            "stable": True,
                            "track_id": obj[
                                "track_id"
                            ],
                            "distance_px": round(
                                dist_r,
                                1
                            )
                        }

                # -----------------------------------------------------
                # Left hand
                # -----------------------------------------------------
                if (
                    lw
                    and lw.get(
                        "confidence",
                        0
                    ) > 0.25
                ):

                    dist_l = (
                        (
                            lw["x"] - cx
                        ) ** 2
                        +
                        (
                            lw["y"] - cy
                        ) ** 2
                    ) ** 0.5

                    near_l = (
                        dist_l < 150.0
                    )

                    if near_l:

                        interaction_signals[
                            f"left_hand_near_{cls_name}"
                        ] = True

                        hand_object_interaction[
                            "left"
                        ][cls_name] = {
                            "near": True,
                            "stable": True,
                            "track_id": obj[
                                "track_id"
                            ],
                            "distance_px": round(
                                dist_l,
                                1
                            )
                        }

        # =============================================================
        # 4. FINAL PERSON 1 OUTPUT
        # =============================================================

        return {
            "frame": frame_num,
            "timestamp": timestamp,
            "objects": objects,
            "pose": pose_dict,
            "hand_object_geometry": {},
            "hand_object_interaction": (
                hand_object_interaction
            ),
            "interaction_signals": (
                interaction_signals
            ),
            "adapter_status": "CONNECTED"
        }

    def _generate_mock_perception_record(
        self,
        frame_num: int,
        timestamp: float
    ) -> Dict[str, Any]:
        """
        Generates a valid mock perception record for development
        and testing when explicit MOCK mode is selected.
        """

        return {
            "frame": frame_num,
            "timestamp": timestamp,

            "objects": [
                {
                    "class": "person",
                    "track_id": 1,
                    "confidence": 0.92,
                    "bbox": [
                        100.0,
                        100.0,
                        500.0,
                        900.0
                    ]
                },
                {
                    "class": "white_container",
                    "track_id": 2,
                    "confidence": 0.88,
                    "bbox": [
                        200.0,
                        600.0,
                        600.0,
                        900.0
                    ]
                }
            ],

            "pose": {
                "person_track_id": 1,

                "left_wrist": {
                    "x": 400.0,
                    "y": 700.0,
                    "confidence": 0.85
                },

                "right_wrist": {
                    "x": 450.0,
                    "y": 720.0,
                    "confidence": 0.88
                }
            },

            "hand_object_geometry": {},

            "hand_object_interaction": {
                "right": {
                    "white_container": {
                        "near": True,
                        "stable": True,
                        "track_id": 2,
                        "distance_px": 15.0
                    }
                }
            },

            "interaction_signals": {
                "right_hand_near_white_container": True
            },

            "adapter_status": "MOCK_ACTIVE"
        }

