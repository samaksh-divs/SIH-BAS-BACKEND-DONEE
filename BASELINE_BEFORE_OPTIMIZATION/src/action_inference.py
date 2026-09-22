"""
Action Inference Layer
Combines single-frame perception signals with temporal history to infer candidate experiment actions.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from src.perception_types import NormalizedFrame
from src.temporal_logic import TemporalHistoryBuffer


@dataclass
class ObservedAction:
    action: str
    confidence: float
    confidence_level: str  # "HIGH", "MEDIUM", "LOW", "UNCERTAIN"
    evidence: List[str]
    start_frame: int
    end_frame: int
    timestamp: float
    involved_objects: List[str]
    wrist_side: str  # "left", "right", "both", "none"
    reason: str


class ActionInferenceEngine:
    """
    Infers state-independent action hypotheses from normalized frames and temporal history.
    """

    KNOWN_ACTIONS = [
        "S01 OPEN_WHITE_BOX",
        "S02 RETRIEVE_RED_BOX",
        "S03 RETRIEVE_YELLOW_BOX",
        "S04 OPEN_RED_BOX",
        "S05 PLANT_TO_WORKPLACE",
        "S06 OPEN_YELLOW_BOX",
        "S07 SPRAY_TO_WORKPLACE",
        "S08 PICK_SPRAY",
        "S09 SPRAY_PLANT",
        "S10 SPRAY_TO_WORKPLACE_AGAIN",
        "S11 PLANT_TO_RED_BOX",
        "S12 SPRAY_TO_YELLOW_BOX",
        "S13 RED_BOX_TO_WHITE_BOX",
        "S14 YELLOW_BOX_TO_WHITE_BOX",
        "S15 CLOSE_WHITE_BOX",
        "S16 COMPLETE"
    ]

    def __init__(self, buffer: Optional[TemporalHistoryBuffer] = None, config_path: Optional[str] = None):
        self.buffer = buffer if buffer is not None else TemporalHistoryBuffer(config_path=config_path)

    def process_frame(self, frame: NormalizedFrame) -> ObservedAction:
        """
        Updates temporal buffer with the new frame and computes the most likely observed action hypothesis.
        """
        self.buffer.add_frame(frame)

        # Gather evidence signals
        evidence: List[str] = []
        involved_objects: List[str] = []
        wrist_side = "none"

        # Check wrist proximity signals
        left_spray = frame.interaction_signals.get("left_hand_near_spray_bottle", False)
        right_spray = frame.interaction_signals.get("right_hand_near_spray_bottle", False)
        left_plant = frame.interaction_signals.get("left_hand_near_plant", False)
        right_plant = frame.interaction_signals.get("right_hand_near_plant", False)
        left_red = frame.interaction_signals.get("left_hand_near_red_box", False)
        right_red = frame.interaction_signals.get("right_hand_near_red_box", False)
        left_yellow = frame.interaction_signals.get("left_hand_near_yellow_box", False)
        right_yellow = frame.interaction_signals.get("right_hand_near_yellow_box", False)
        left_white = frame.interaction_signals.get("left_hand_near_white_container", False)
        right_white = frame.interaction_signals.get("right_hand_near_white_container", False)

        if left_spray or left_plant or left_red or left_yellow or left_white:
            wrist_side = "left"
        if right_spray or right_plant or right_red or right_yellow or right_white:
            wrist_side = "both" if wrist_side == "left" else "right"

        # Check temporal persistence over sliding window
        spray_persist = max(
            self.buffer.get_signal_persistence("left_hand_near_spray_bottle"),
            self.buffer.get_signal_persistence("right_hand_near_spray_bottle")
        )
        plant_persist = max(
            self.buffer.get_signal_persistence("left_hand_near_plant"),
            self.buffer.get_signal_persistence("right_hand_near_plant")
        )
        red_persist = max(
            self.buffer.get_signal_persistence("left_hand_near_red_box"),
            self.buffer.get_signal_persistence("right_hand_near_red_box")
        )
        yellow_persist = max(
            self.buffer.get_signal_persistence("left_hand_near_yellow_box"),
            self.buffer.get_signal_persistence("right_hand_near_yellow_box")
        )
        white_persist = max(
            self.buffer.get_signal_persistence("left_hand_near_white_container"),
            self.buffer.get_signal_persistence("right_hand_near_white_container")
        )

        # Check object displacements over temporal window
        spray_disp = self.buffer.get_object_displacement("spray_bottle")
        plant_disp = self.buffer.get_object_displacement("plant")
        red_disp = self.buffer.get_object_displacement("red_box")
        yellow_disp = self.buffer.get_object_displacement("yellow_box")
        wrist_disp = self.buffer.get_wrist_displacement(hand="right" if right_spray else "left")

        # -------------------------------------------------------------
        # Action Hypothesis Heuristics
        # -------------------------------------------------------------

        # 1. SPRAY_PLANT: Hand near spray bottle AND hand near plant persistently
        if (left_spray or right_spray) and (left_plant or right_plant):
            evidence.append("hand near spray_bottle")
            evidence.append("hand near plant")
            evidence.append(f"spray persistence: {spray_persist} frames")
            evidence.append(f"plant persistence: {plant_persist} frames")
            involved_objects = ["spray_bottle", "plant"]

            conf = 0.90 if (spray_persist >= 3 and plant_persist >= 3) else 0.65
            conf_level = "HIGH" if conf >= 0.85 else "MEDIUM"
            return ObservedAction(
                action="SPRAY_PLANT",
                confidence=conf,
                confidence_level=conf_level,
                evidence=evidence,
                start_frame=max(1, frame.frame - len(self.buffer.frames) + 1),
                end_frame=frame.frame,
                timestamp=frame.timestamp,
                involved_objects=involved_objects,
                wrist_side=wrist_side,
                reason="Simultaneous hand proximity to spray_bottle and plant"
            )

        # 2. PICK_SPRAY vs SPRAY_TO_WORKPLACE vs SPRAY_TO_WORKPLACE_AGAIN
        if (left_spray or right_spray) or self.buffer.is_object_recently_present("spray_bottle"):
            if left_spray or right_spray:
                involved_objects = ["spray_bottle"]
                evidence.append("hand near spray_bottle")

                if wrist_disp and wrist_disp["is_moving_up"]:
                    evidence.append(f"wrist lifting upward (dy={wrist_disp['dy']:.1f}px)")
                    return ObservedAction(
                        action="PICK_SPRAY",
                        confidence=0.88,
                        confidence_level="HIGH",
                        evidence=evidence,
                        start_frame=max(1, frame.frame - len(self.buffer.frames) + 1),
                        end_frame=frame.frame,
                        timestamp=frame.timestamp,
                        involved_objects=involved_objects,
                        wrist_side=wrist_side,
                        reason="Hand near spray_bottle with upward lifting trajectory"
                    )

                if wrist_disp and wrist_disp["is_moving_down"]:
                    evidence.append(f"wrist lowering downward (dy={wrist_disp['dy']:.1f}px)")
                    return ObservedAction(
                        action="SPRAY_TO_WORKPLACE_AGAIN",
                        confidence=0.82,
                        confidence_level="MEDIUM",
                        evidence=evidence,
                        start_frame=max(1, frame.frame - len(self.buffer.frames) + 1),
                        end_frame=frame.frame,
                        timestamp=frame.timestamp,
                        involved_objects=involved_objects,
                        wrist_side=wrist_side,
                        reason="Hand near spray_bottle with downward setting trajectory"
                    )

                # General spray handling when stationary or persistent
                if spray_persist >= 3:
                    evidence.append(f"spray bottle interaction persistence ({spray_persist} frames)")
                    return ObservedAction(
                        action="PICK_SPRAY",
                        confidence=0.75,
                        confidence_level="MEDIUM",
                        evidence=evidence,
                        start_frame=max(1, frame.frame - len(self.buffer.frames) + 1),
                        end_frame=frame.frame,
                        timestamp=frame.timestamp,
                        involved_objects=involved_objects,
                        wrist_side=wrist_side,
                        reason="Persistent hand near spray_bottle"
                    )

        # 3. PLANT_TO_WORKPLACE / PLANT_TO_RED_BOX
        if (left_plant or right_plant) or self.buffer.is_object_recently_present("plant"):
            if left_plant or right_plant:
                involved_objects = ["plant"]
                evidence.append("hand near plant")

                if plant_disp:
                    evidence.append(f"plant displacement distance={plant_disp['distance_px']:.1f}px")
                    if plant_disp["is_moving_left"] or plant_disp["is_moving_down"]:
                        return ObservedAction(
                            action="PLANT_TO_WORKPLACE",
                            confidence=0.85,
                            confidence_level="HIGH",
                            evidence=evidence,
                            start_frame=max(1, frame.frame - len(self.buffer.frames) + 1),
                            end_frame=frame.frame,
                            timestamp=frame.timestamp,
                            involved_objects=involved_objects,
                            wrist_side=wrist_side,
                            reason="Hand near plant with movement toward workplace"
                        )
                    elif plant_disp["is_moving_right"] or plant_disp["is_moving_up"]:
                        return ObservedAction(
                            action="PLANT_TO_RED_BOX",
                            confidence=0.80,
                            confidence_level="MEDIUM",
                            evidence=evidence,
                            start_frame=max(1, frame.frame - len(self.buffer.frames) + 1),
                            end_frame=frame.frame,
                            timestamp=frame.timestamp,
                            involved_objects=involved_objects,
                            wrist_side=wrist_side,
                            reason="Hand near plant with movement toward red box"
                        )

                if plant_persist >= 3:
                    return ObservedAction(
                        action="PLANT_TO_WORKPLACE",
                        confidence=0.70,
                        confidence_level="MEDIUM",
                        evidence=evidence,
                        start_frame=max(1, frame.frame - len(self.buffer.frames) + 1),
                        end_frame=frame.frame,
                        timestamp=frame.timestamp,
                        involved_objects=["plant"],
                        wrist_side=wrist_side,
                        reason="Hand interacting with plant"
                    )

        # 4. RED_BOX actions (RETRIEVE_RED_BOX, OPEN_RED_BOX, RED_BOX_TO_WHITE_BOX)
        if (left_red or right_red) or (red_disp and red_disp["distance_px"] > 15.0):
            involved_objects = ["red_box"]
            evidence.append(f"red box interaction persistence: {red_persist} frames")

            if red_disp and red_disp["distance_px"] > 20.0:
                evidence.append(f"red box displacement: {red_disp['distance_px']:.1f}px")
                if red_disp["is_moving_left"] or red_disp["is_moving_down"]:
                    return ObservedAction(
                        action="RETRIEVE_RED_BOX",
                        confidence=0.86,
                        confidence_level="HIGH",
                        evidence=evidence,
                        start_frame=max(1, frame.frame - len(self.buffer.frames) + 1),
                        end_frame=frame.frame,
                        timestamp=frame.timestamp,
                        involved_objects=involved_objects,
                        wrist_side=wrist_side,
                        reason="Red box displacement out of white container"
                    )
                else:
                    return ObservedAction(
                        action="RED_BOX_TO_WHITE_BOX",
                        confidence=0.82,
                        confidence_level="MEDIUM",
                        evidence=evidence,
                        start_frame=max(1, frame.frame - len(self.buffer.frames) + 1),
                        end_frame=frame.frame,
                        timestamp=frame.timestamp,
                        involved_objects=involved_objects,
                        wrist_side=wrist_side,
                        reason="Red box displacement toward white container"
                    )

            if red_persist >= 3:
                return ObservedAction(
                    action="OPEN_RED_BOX",
                    confidence=0.78,
                    confidence_level="MEDIUM",
                    evidence=evidence,
                    start_frame=max(1, frame.frame - len(self.buffer.frames) + 1),
                    end_frame=frame.frame,
                    timestamp=frame.timestamp,
                    involved_objects=involved_objects,
                    wrist_side=wrist_side,
                    reason="Persistent hand interaction with red_box"
                )

        # 5. YELLOW_BOX actions (RETRIEVE_YELLOW_BOX, OPEN_YELLOW_BOX, YELLOW_BOX_TO_WHITE_BOX)
        if (left_yellow or right_yellow) or (yellow_disp and yellow_disp["distance_px"] > 15.0):
            involved_objects = ["yellow_box"]
            evidence.append(f"yellow box interaction persistence: {yellow_persist} frames")

            if yellow_disp and yellow_disp["distance_px"] > 20.0:
                evidence.append(f"yellow box displacement: {yellow_disp['distance_px']:.1f}px")
                if yellow_disp["is_moving_left"] or yellow_disp["is_moving_down"]:
                    return ObservedAction(
                        action="RETRIEVE_YELLOW_BOX",
                        confidence=0.86,
                        confidence_level="HIGH",
                        evidence=evidence,
                        start_frame=max(1, frame.frame - len(self.buffer.frames) + 1),
                        end_frame=frame.frame,
                        timestamp=frame.timestamp,
                        involved_objects=involved_objects,
                        wrist_side=wrist_side,
                        reason="Yellow box displacement out of white container"
                    )
                else:
                    return ObservedAction(
                        action="YELLOW_BOX_TO_WHITE_BOX",
                        confidence=0.82,
                        confidence_level="MEDIUM",
                        evidence=evidence,
                        start_frame=max(1, frame.frame - len(self.buffer.frames) + 1),
                        end_frame=frame.frame,
                        timestamp=frame.timestamp,
                        involved_objects=involved_objects,
                        wrist_side=wrist_side,
                        reason="Yellow box displacement toward white container"
                    )

            if yellow_persist >= 3:
                return ObservedAction(
                    action="OPEN_YELLOW_BOX",
                    confidence=0.78,
                    confidence_level="MEDIUM",
                    evidence=evidence,
                    start_frame=max(1, frame.frame - len(self.buffer.frames) + 1),
                    end_frame=frame.frame,
                    timestamp=frame.timestamp,
                    involved_objects=involved_objects,
                    wrist_side=wrist_side,
                    reason="Persistent hand interaction with yellow_box"
                )

        # 6. WHITE_CONTAINER actions (OPEN_WHITE_BOX / CLOSE_WHITE_BOX)
        if left_white or right_white:
            involved_objects = ["white_container"]
            evidence.append("hand near white_container")
            evidence.append(f"white container persistence: {white_persist} frames")

            if white_persist >= 3:
                return ObservedAction(
                    action="OPEN_WHITE_BOX",
                    confidence=0.80,
                    confidence_level="MEDIUM",
                    evidence=evidence,
                    start_frame=max(1, frame.frame - len(self.buffer.frames) + 1),
                    end_frame=frame.frame,
                    timestamp=frame.timestamp,
                    involved_objects=involved_objects,
                    wrist_side=wrist_side,
                    reason="Persistent hand interaction with white_container"
                )

        # Default fallback when no strong action evidence is present
        return ObservedAction(
            action="UNCERTAIN",
            confidence=0.20,
            confidence_level="UNCERTAIN",
            evidence=["No strong object proximity or displacement signals"],
            start_frame=frame.frame,
            end_frame=frame.frame,
            timestamp=frame.timestamp,
            involved_objects=[],
            wrist_side="none",
            reason="Insufficient evidence to classify action hypothesis"
        )
