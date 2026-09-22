"""
Temporal Logic Layer
Maintains sliding windows, object track history, wrist trajectory, interaction persistence,
and grace periods for temporary CV detection gaps.
"""
import json
import math
import os
from collections import deque
from typing import Dict, List, Optional, Tuple, Any

from src.perception_types import NormalizedFrame, DetectedObject, WristPose


class TemporalHistoryBuffer:
    """
    Sliding window buffer for accumulating perception frames and calculating temporal heuristics.
    """

    def __init__(self, config_path: Optional[str] = None):
        # Default fallback thresholds
        self.window_size = 15
        self.min_evidence_frames = 5
        self.movement_threshold_px = 20.0
        self.interaction_persistence_frames = 3
        self.detection_gap_grace_period_frames = 8

        if config_path and os.path.exists(config_path):
            self.load_config(config_path)

        self.frames: deque[NormalizedFrame] = deque(maxlen=self.window_size)
        self.class_positions: Dict[str, deque[Tuple[float, float, int]]] = {}
        self.track_positions: Dict[int, deque[Tuple[float, float, int]]] = {}
        self.wrist_positions: Dict[str, deque[Tuple[float, float, int]]] = {
            "left": deque(maxlen=self.window_size),
            "right": deque(maxlen=self.window_size)
        }

    def load_config(self, config_path: str) -> None:
        """Loads configurable threshold values from json file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
            t_cfg = cfg.get("temporal", {})
            self.window_size = t_cfg.get("window_size", self.window_size)
            self.min_evidence_frames = t_cfg.get("min_evidence_frames", self.min_evidence_frames)
            self.movement_threshold_px = t_cfg.get("movement_threshold_px", self.movement_threshold_px)
            self.interaction_persistence_frames = t_cfg.get(
                "interaction_persistence_frames", self.interaction_persistence_frames
            )
            self.detection_gap_grace_period_frames = t_cfg.get(
                "detection_gap_grace_period_frames", self.detection_gap_grace_period_frames
            )

    def add_frame(self, frame: NormalizedFrame) -> None:
        """Appends a new frame to the temporal history buffer and updates track position queues."""
        self.frames.append(frame)

        # Update object positions
        for obj in frame.objects:
            # Update class position history
            if obj.class_name not in self.class_positions:
                self.class_positions[obj.class_name] = deque(maxlen=self.window_size)
            self.class_positions[obj.class_name].append((obj.center[0], obj.center[1], frame.frame))

            # Update track position history
            if obj.track_id is not None:
                if obj.track_id not in self.track_positions:
                    self.track_positions[obj.track_id] = deque(maxlen=self.window_size)
                self.track_positions[obj.track_id].append((obj.center[0], obj.center[1], frame.frame))

        # Update wrist positions
        if frame.pose.left_wrist.x is not None and frame.pose.left_wrist.y is not None:
            self.wrist_positions["left"].append(
                (frame.pose.left_wrist.x, frame.pose.left_wrist.y, frame.frame)
            )

        if frame.pose.right_wrist.x is not None and frame.pose.right_wrist.y is not None:
            self.wrist_positions["right"].append(
                (frame.pose.right_wrist.x, frame.pose.right_wrist.y, frame.frame)
            )

    def is_object_recently_present(self, class_name: str, grace_period_frames: Optional[int] = None) -> bool:
        """
        Returns True if the object was detected in any of the recent N frames.
        Helps tolerate temporary CV detection gaps.
        """
        grace = grace_period_frames if grace_period_frames is not None else self.detection_gap_grace_period_frames
        recent_frames = list(self.frames)[-grace:]
        for f in recent_frames:
            if f.get_object_by_class(class_name) is not None:
                return True
        return False

    def get_signal_persistence(self, signal_name: str, window_len: Optional[int] = None) -> int:
        """Counts how many frames in the recent window had the specified signal == True."""
        w_len = window_len if window_len is not None else len(self.frames)
        recent_frames = list(self.frames)[-w_len:]
        return sum(1 for f in recent_frames if f.interaction_signals.get(signal_name, False))

    def get_hand_near_persistence(self, object_class: str, hand: str = "any", window_len: Optional[int] = None) -> int:
        """Counts frames in recent window where hand was near the given object class."""
        w_len = window_len if window_len is not None else len(self.frames)
        recent_frames = list(self.frames)[-w_len:]
        return sum(1 for f in recent_frames if f.is_hand_near(object_class, hand=hand))

    def get_object_displacement(self, class_name: str, window_len: Optional[int] = None) -> Optional[Dict[str, float]]:
        """
        Calculates position displacement (dx, dy, total_distance, dy_upward) for an object class
        between the oldest and newest frame in the specified window.
        """
        if class_name not in self.class_positions or len(self.class_positions[class_name]) < 2:
            return None

        pos_queue = self.class_positions[class_name]
        w_len = window_len if window_len is not None else len(pos_queue)
        sample = list(pos_queue)[-w_len:]
        if len(sample) < 2:
            return None

        first = sample[0]
        last = sample[-1]
        dx = last[0] - first[0]
        dy = last[1] - first[1]  # positive = moving down in pixel space
        dist = math.hypot(dx, dy)
        frame_diff = max(1, last[2] - first[2])

        return {
            "dx": dx,
            "dy": dy,
            "distance_px": dist,
            "speed_px_per_frame": dist / frame_diff,
            "is_moving_up": dy < -5.0,
            "is_moving_down": dy > 5.0,
            "is_moving_left": dx < -5.0,
            "is_moving_right": dx > 5.0,
            "frame_delta": frame_diff
        }

    def get_wrist_displacement(self, hand: str = "right", window_len: Optional[int] = None) -> Optional[Dict[str, float]]:
        """Calculates wrist displacement over the recent window."""
        w_queue = self.wrist_positions.get(hand, deque())
        if len(w_queue) < 2:
            return None

        w_len = window_len if window_len is not None else len(w_queue)
        sample = list(w_queue)[-w_len:]
        if len(sample) < 2:
            return None

        first = sample[0]
        last = sample[-1]
        dx = last[0] - first[0]
        dy = last[1] - first[1]
        dist = math.hypot(dx, dy)
        frame_diff = max(1, last[2] - first[2])

        return {
            "dx": dx,
            "dy": dy,
            "distance_px": dist,
            "speed_px_per_frame": dist / frame_diff,
            "is_moving_up": dy < -5.0,
            "is_moving_down": dy > 5.0
        }

    def clear(self) -> None:
        """Clears all stored history."""
        self.frames.clear()
        self.class_positions.clear()
        self.track_positions.clear()
        self.wrist_positions["left"].clear()
        self.wrist_positions["right"].clear()
