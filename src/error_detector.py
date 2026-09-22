"""
Procedural Error Detection Layer
Classifies procedural errors, debounces alerts, and categorizes violations.
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any


class ErrorCategory:
    OUT_OF_SEQUENCE = "OUT_OF_SEQUENCE"
    SKIPPED_STEP = "SKIPPED_STEP"
    WRONG_OBJECT = "WRONG_OBJECT"
    WRONG_TARGET = "WRONG_TARGET"
    INACTION = "INACTION"
    PERCEPTION_UNCERTAIN = "PERCEPTION_UNCERTAIN"
    DETECTION_GAP = "DETECTION_GAP"


@dataclass
class ErrorEvent:
    error_type: str
    message: str
    expected_action: str
    observed_action: str
    frame: int
    timestamp: float
    is_debounced: bool = False


class ProceduralErrorDetector:
    """
    Evaluates observed action against expected state to detect procedural errors.
    Includes debouncing logic to prevent repeating identical error alerts every frame.
    """

    def __init__(self, cooldown_frames: int = 15):
        self.cooldown_frames = cooldown_frames
        self.last_error_type: Optional[str] = None
        self.last_error_frame: int = -999

    def evaluate(
        self,
        expected_state_id: str,
        expected_action: str,
        observed_action: str,
        confidence: float,
        confidence_level: str,
        step_index_map: Dict[str, int],
        frame: int,
        timestamp: float
    ) -> Optional[ErrorEvent]:
        """
        Classifies whether an observed action represents a procedural error.
        Returns an ErrorEvent if a violation occurs, else None.
        """
        if observed_action in ("IDLE", "UNCERTAIN") or confidence_level == "UNCERTAIN":
            return None  # Weak perception handled separately as PERCEPTION_UNCERTAIN

        if observed_action == expected_action:
            # Reset error tracking when correct action is observed
            self.last_error_type = None
            return None

        # Check if observed action belongs to another state in the sequence
        exp_idx = step_index_map.get(expected_action, -1)
        obs_idx = step_index_map.get(observed_action, -1)

        error_type = None
        message = ""

        if obs_idx > exp_idx:
            if obs_idx == exp_idx + 1:
                error_type = ErrorCategory.SKIPPED_STEP
                message = f"Warning: Step '{expected_action}' appears to have been skipped."
            else:
                error_type = ErrorCategory.OUT_OF_SEQUENCE
                message = f"Warning: Out of sequence action '{observed_action}' detected. Expected '{expected_action}'."
        elif obs_idx >= 0 and obs_idx < exp_idx:
            error_type = ErrorCategory.OUT_OF_SEQUENCE
            message = f"Warning: Repeated previous step '{observed_action}'. Expected '{expected_action}'."
        else:
            # Action recognized but mismatched object context
            error_type = ErrorCategory.WRONG_OBJECT
            message = f"Warning: Incorrect object interaction detected. Expected '{expected_action}'."

        if error_type is None:
            return None

        # Apply debouncing logic
        is_debounced = False
        if error_type == self.last_error_type and (frame - self.last_error_frame) < self.cooldown_frames:
            is_debounced = True

        self.last_error_type = error_type
        self.last_error_frame = frame

        return ErrorEvent(
            error_type=error_type,
            message=message,
            expected_action=expected_action,
            observed_action=observed_action,
            frame=frame,
            timestamp=timestamp,
            is_debounced=is_debounced
        )

    def reset(self) -> None:
        """Resets error detector memory."""
        self.last_error_type = None
        self.last_error_frame = -999
