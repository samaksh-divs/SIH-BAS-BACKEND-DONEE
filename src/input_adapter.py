"""
Input Adapter Layer
Parses frame-by-frame JSON/dict perception outputs from Person 1 into NormalizedFrame instances.
"""
import json
from typing import Dict, Any, Generator, Optional, Union
from src.perception_types import (
    NormalizedFrame,
    DetectedObject,
    PoseData,
    WristPose,
    HandObjectInteraction,
    InteractionDetail
)


class Person1Adapter:
    """
    Adapter class for validating and converting Person 1 visual output records
    into normalized dataclass instances.
    """

    KNOWN_CLASSES = {
        "person",
        "white_container",
        "red_box",
        "yellow_box",
        "plant",
        "spray_bottle"
    }

    KNOWN_SIGNALS = [
        "left_hand_near_spray_bottle",
        "right_hand_near_spray_bottle",
        "left_hand_near_plant",
        "right_hand_near_plant",
        "left_hand_near_red_box",
        "right_hand_near_red_box",
        "left_hand_near_yellow_box",
        "right_hand_near_yellow_box",
        "left_hand_near_white_container",
        "right_hand_near_white_container"
    ]

    @classmethod
    def parse_record(cls, data: Union[str, Dict[str, Any]]) -> NormalizedFrame:
        """
        Parses a single JSON string or dict into a NormalizedFrame object.
        """
        if isinstance(data, str):
            record = json.loads(data.strip())
        elif isinstance(data, dict):
            record = data
        else:
            raise TypeError(f"Expected str or dict, got {type(data)}")

        frame_num = int(record.get("frame", 0))
        timestamp = float(record.get("timestamp", 0.0))

        # Parse detected objects
        raw_objects = record.get("objects", [])
        parsed_objects = []
        for raw_obj in raw_objects:
            cls_name = str(raw_obj.get("class", ""))
            track_id = raw_obj.get("track_id")
            if track_id is not None:
                track_id = int(track_id)
            conf = float(raw_obj.get("confidence", 0.0))
            bbox_raw = raw_obj.get("bbox", [0.0, 0.0, 0.0, 0.0])
            bbox = (float(bbox_raw[0]), float(bbox_raw[1]), float(bbox_raw[2]), float(bbox_raw[3]))
            center_x = (bbox[0] + bbox[2]) / 2.0
            center_y = (bbox[1] + bbox[3]) / 2.0

            parsed_objects.append(
                DetectedObject(
                    class_name=cls_name,
                    track_id=track_id,
                    confidence=conf,
                    bbox=bbox,
                    center=(center_x, center_y)
                )
            )

        # Parse pose
        raw_pose = record.get("pose", {})
        person_track_id = raw_pose.get("person_track_id")
        if person_track_id is not None:
            person_track_id = int(person_track_id)

        left_wrist_raw = raw_pose.get("left_wrist", {})
        left_wrist = WristPose(
            x=float(left_wrist_raw["x"]) if left_wrist_raw.get("x") is not None else None,
            y=float(left_wrist_raw["y"]) if left_wrist_raw.get("y") is not None else None,
            confidence=float(left_wrist_raw.get("confidence", 0.0))
        )

        right_wrist_raw = raw_pose.get("right_wrist", {})
        right_wrist = WristPose(
            x=float(right_wrist_raw["x"]) if right_wrist_raw.get("x") is not None else None,
            y=float(right_wrist_raw["y"]) if right_wrist_raw.get("y") is not None else None,
            confidence=float(right_wrist_raw.get("confidence", 0.0))
        )

        pose = PoseData(
            person_track_id=person_track_id,
            left_wrist=left_wrist,
            right_wrist=right_wrist
        )

        # Parse hand-object interaction details
        raw_hoi = record.get("hand_object_interaction", {})
        left_hoi = cls._parse_hand_interactions(raw_hoi.get("left", {}))
        right_hoi = cls._parse_hand_interactions(raw_hoi.get("right", {}))
        hoi = HandObjectInteraction(left=left_hoi, right=right_hoi)

        # Parse interaction signals
        raw_signals = record.get("interaction_signals", {})
        interaction_signals = {
            sig_key: bool(raw_signals.get(sig_key, False))
            for sig_key in cls.KNOWN_SIGNALS
        }

        return NormalizedFrame(
            frame=frame_num,
            timestamp=timestamp,
            objects=parsed_objects,
            pose=pose,
            hand_object_interaction=hoi,
            interaction_signals=interaction_signals
        )

    @classmethod
    def _parse_hand_interactions(cls, raw_dict: Dict[str, Any]) -> Dict[str, InteractionDetail]:
        res = {}
        for obj_key, detail in raw_dict.items():
            if isinstance(detail, dict):
                tid = detail.get("track_id")
                dist = detail.get("distance_px")
                res[obj_key] = InteractionDetail(
                    near=bool(detail.get("near", False)),
                    stable=bool(detail.get("stable", False)),
                    track_id=int(tid) if tid is not None else None,
                    distance_px=float(dist) if dist is not None else None
                )
        return res

    @classmethod
    def stream_file(cls, filepath: str) -> Generator[NormalizedFrame, None, None]:
        """
        Reads a JSONL file line-by-line and yields NormalizedFrame objects.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    yield cls.parse_record(line_str)
