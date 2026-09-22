"""
Perception Types Definition
Normalized internal data structure representing frame-level output from Person 1 CV pipeline.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


@dataclass
class WristPose:
    x: Optional[float] = None
    y: Optional[float] = None
    confidence: float = 0.0


@dataclass
class PoseData:
    person_track_id: Optional[int] = None
    left_wrist: WristPose = field(default_factory=WristPose)
    right_wrist: WristPose = field(default_factory=WristPose)


@dataclass
class DetectedObject:
    class_name: str
    track_id: Optional[int]
    confidence: float
    bbox: Tuple[float, float, float, float]  # [x1, y1, x2, y2]
    center: Tuple[float, float]  # [center_x, center_y]


@dataclass
class InteractionDetail:
    near: bool = False
    stable: bool = False
    track_id: Optional[int] = None
    distance_px: Optional[float] = None


@dataclass
class HandObjectInteraction:
    left: Dict[str, InteractionDetail] = field(default_factory=dict)
    right: Dict[str, InteractionDetail] = field(default_factory=dict)


@dataclass
class NormalizedFrame:
    frame: int
    timestamp: float
    objects: List[DetectedObject]
    pose: PoseData
    hand_object_interaction: HandObjectInteraction
    interaction_signals: Dict[str, bool]

    def get_object_by_class(self, class_name: str) -> Optional[DetectedObject]:
        """Utility to get the highest confidence detected object of a specific class."""
        matching = [obj for obj in self.objects if obj.class_name == class_name]
        if not matching:
            return None
        return max(matching, key=lambda obj: obj.confidence)

    def is_hand_near(self, object_class: str, hand: str = "any") -> bool:
        """Utility to check if hand is near a target object via interaction_signals or interaction details."""
        if hand in ("left", "any"):
            sig_left = self.interaction_signals.get(f"left_hand_near_{object_class}", False)
            detail_left = self.hand_object_interaction.left.get(object_class)
            if sig_left or (detail_left and detail_left.near):
                return True
        if hand in ("right", "any"):
            sig_right = self.interaction_signals.get(f"right_hand_near_{object_class}", False)
            detail_right = self.hand_object_interaction.right.get(object_class)
            if sig_right or (detail_right and detail_right.near):
                return True
        return False
