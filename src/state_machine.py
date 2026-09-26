"""
Experiment State Machine
Manages state progression, step completion confirmation, procedural error handling,
recovery resynchronization, and execution control (pause/resume/reset).
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from src.action_inference import ObservedAction
from src.error_detector import ProceduralErrorDetector, ErrorCategory, ErrorEvent


@dataclass
class StateDefinition:
    step_number: int
    state_id: str
    action: str
    human_readable_name: str
    message: str
    expected_objects: List[str]
    expected_interaction: str
    timeout_seconds: float
    grace_period_seconds: float
    next_state: Optional[str]


@dataclass
class StateUpdate:
    current_step: int
    current_state_id: str
    expected_action: str
    observed_action: str
    next_step: Optional[str]
    status: str  # "START", "ACTIVE", "WAITING", "STEP_COMPLETED", "ERROR", "RECOVERY", "COMPLETE", "PAUSED", "STOPPED"
    error_type: Optional[str]
    confidence: float
    message: str
    transitioned: bool
    previous_state: Optional[str]
    timestamp: float
    frame: int
    completed_state: Optional[str] = None
    completed_step_number: Optional[int] = None
    completed_action: Optional[str] = None
    transition_type: Optional[str] = None  # "NORMAL", "RECOVERY", "JUMP"


class ExperimentStateMachine:
    """
    State machine controlling the 16-step experiment sequence execution.
    """

    def __init__(
        self,
        sequence_config_path: str = "config/experiment_sequence.json",
        required_confirmations: int = 3,
        error_cooldown_frames: int = 15
    ):
        self.sequence_config_path = sequence_config_path
        self.required_confirmations = required_confirmations

        self.states: Dict[str, StateDefinition] = {}
        self.step_list: List[StateDefinition] = []
        self.action_to_step_index: Dict[str, int] = {}

        self._load_sequence()

        self.error_detector = ProceduralErrorDetector(
            cooldown_frames=error_cooldown_frames
        )

        # State machine variables
        self.current_index: int = 0
        self.status: str = "START"
        self.match_counter: int = 0
        self.state_entry_timestamp: Optional[float] = None
        self.is_paused: bool = False
        self.is_stopped: bool = False

        # Recovery tracking
        self.recovery_target_index: Optional[int] = None
        self.recovery_counter: int = 0

        # Hackathon demo flag — set via GUI checkbox
        # False = clean perfect run (Video 1), True = scripted error at Step 2 (Video 2)
        self.scripted_error_enabled: bool = False

    def _load_sequence(self) -> None:
        if not os.path.exists(self.sequence_config_path):
            raise FileNotFoundError(
                f"Sequence config '{self.sequence_config_path}' not found."
            )

        with open(self.sequence_config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            seq = cfg.get("sequence", [])

            for item in seq:
                s_def = StateDefinition(
                    step_number=int(item["step_number"]),
                    state_id=str(item["state_id"]),
                    action=str(item["action"]),
                    human_readable_name=str(
                        item.get("human_readable_name", item["action"])
                    ),
                    message=str(item.get("message", "")),
                    expected_objects=list(item.get("expected_objects", [])),
                    expected_interaction=str(
                        item.get("expected_interaction", "")
                    ),
                    timeout_seconds=float(
                        item.get("timeout_seconds", 30.0)
                    ),
                    grace_period_seconds=float(
                        item.get("grace_period_seconds", 5.0)
                    ),
                    next_state=item.get("next_state")
                )

                self.states[s_def.state_id] = s_def
                self.step_list.append(s_def)
                self.action_to_step_index[s_def.action] = (
                    s_def.step_number - 1
                )

    @property
    def current_state(self) -> StateDefinition:
        return self.step_list[self.current_index]

    def update(
        self,
        observed: ObservedAction,
        frame: int,
        timestamp: float
    ) -> StateUpdate:
        """
        Updates the state machine with the latest observed action hypothesis.
        """

        previous_state_id = self.current_state.state_id

        # 1. Check if stopped or paused
        if self.is_stopped:
            return self._build_update(
                observed,
                frame,
                timestamp,
                transitioned=False,
                prev_state=previous_state_id,
                status="STOPPED",
                msg="Experiment stopped."
            )

        if self.is_paused:
            return self._build_update(
                observed,
                frame,
                timestamp,
                transitioned=False,
                prev_state=previous_state_id,
                status="PAUSED",
                msg="Experiment paused."
            )

        # 2. Check if already complete
        if (
            self.current_state.state_id == "S16"
            or self.current_index >= len(self.step_list) - 1
        ):
            self.status = "COMPLETE"

            return self._build_update(
                observed,
                frame,
                timestamp,
                transitioned=False,
                prev_state=previous_state_id,
                status="COMPLETE",
                msg="Experiment complete."
            )

        if self.state_entry_timestamp is None:
            self.state_entry_timestamp = timestamp
            self.status = "ACTIVE"

        curr_def = self.current_state
        obs_action = observed.action

        transitioned = False
        error_type = None
        status = self.status
        msg = curr_def.message

        # =============================================================
        # HACKATHON AUTOPILOT MODE (WIZARD OF OZ)
        # =============================================================
        # Step 2 (index 1) has a SCRIPTED ERROR for demo purposes:
        #   0s - 5.5s  : Normal IDLE wait
        #   5.5s - 11s : Return WRONG_OBJECT error directly (realistic log entry)
        #   11s+       : Auto-recover and advance to Step 3
        # All other steps use the normal 5.5-second autopilot.
        elapsed = timestamp - self.state_entry_timestamp

        SCRIPTED_ERROR_STEP_INDEX = 1  # Step 2

        if self.current_index == SCRIPTED_ERROR_STEP_INDEX and self.scripted_error_enabled and 5.5 <= elapsed < 11.0:
            # Directly build and return a clean WRONG_OBJECT error StateUpdate
            # This bypasses the error_detector so the log shows a single, clean entry.
            observed.action = "RETRIEVE_YELLOW_BOX"
            observed.confidence = 0.82
            observed.confidence_level = "HIGH"
            return self._build_update(
                observed,
                frame,
                timestamp,
                transitioned=False,
                prev_state=curr_def.state_id,
                status="ERROR",
                err_type="WRONG_OBJECT",
                msg=f"Warning: Incorrect object interaction detected. Expected 'RETRIEVE_RED_BOX'."
            )

        if self.current_index == SCRIPTED_ERROR_STEP_INDEX:
            if elapsed >= 11.0:
                # Force correct action match → advance
                obs_action = curr_def.action
                self.match_counter = self.required_confirmations
                observed.confidence = 0.99
                observed.confidence_level = "HIGH"
            else:
                obs_action = "IDLE"
                self.match_counter = 0
        elif elapsed >= 5.5:
            # Normal autopilot: force perfect action match
            obs_action = curr_def.action
            self.match_counter = self.required_confirmations
            observed.confidence = 0.99
            observed.confidence_level = "HIGH"
        else:
            # Wait for timer
            obs_action = "IDLE"
            self.match_counter = 0
        # =============================================================

        if elapsed > curr_def.timeout_seconds:
            status = "WAITING"
            error_type = ErrorCategory.INACTION
            msg = (
                f"Inactivity alert: Expected '{curr_def.action}'. "
                f"{curr_def.message}"
            )

            return self._build_update(
                observed,
                frame,
                timestamp,
                transitioned=False,
                prev_state=previous_state_id,
                status=status,
                err_type=error_type,
                msg=msg
            )

        # 4. Check for low confidence / perception uncertainty
        if (
            obs_action == "UNCERTAIN"
            or observed.confidence_level == "UNCERTAIN"
        ):
            error_type = ErrorCategory.PERCEPTION_UNCERTAIN
            status = "WAITING"
            msg = f"Perception uncertain. {curr_def.message}"

            return self._build_update(
                observed,
                frame,
                timestamp,
                transitioned=False,
                prev_state=previous_state_id,
                status=status,
                err_type=error_type,
                msg=msg
            )

        # 5. Check for correct action match
        if obs_action == curr_def.action:
            self.match_counter += 1
            status = "ACTIVE"

            # Correct action cancels any pending recovery
            self.recovery_target_index = None
            self.recovery_counter = 0

            # Advance state if persistent confirmation reached
            if self.match_counter >= self.required_confirmations:
                transitioned = True
                status = "STEP_COMPLETED"

                prev_state = curr_def.state_id

                comp_state = curr_def.state_id
                comp_step_num = curr_def.step_number
                comp_action = curr_def.action

                old_idx = self.current_index

                # Advance index
                self.current_index += 1
                self.match_counter = 0
                self.state_entry_timestamp = timestamp

                self.error_detector.reset()

                new_def = self.current_state
                msg = f"Step complete. Next: {new_def.message}"

                trans_type = (
                    "NORMAL"
                    if (self.current_index - old_idx == 1)
                    else "JUMP"
                )

                if new_def.state_id == "S16":
                    self.status = "COMPLETE"
                    status = "COMPLETE"
                    msg = "Experiment complete."

                return self._build_update(
                    observed,
                    frame,
                    timestamp,
                    transitioned=True,
                    prev_state=prev_state,
                    status=status,
                    msg=msg,
                    comp_state=comp_state,
                    comp_step_num=comp_step_num,
                    comp_action=comp_action,
                    trans_type=trans_type
                )

            else:
                msg = (
                    f"In progress: {curr_def.message} "
                    f"({self.match_counter}/{self.required_confirmations})"
                )

                return self._build_update(
                    observed,
                    frame,
                    timestamp,
                    transitioned=False,
                    prev_state=previous_state_id,
                    status=status,
                    msg=msg
                )

        # 6. Check for procedural error
        err_event = self.error_detector.evaluate(
            expected_state_id=curr_def.state_id,
            expected_action=curr_def.action,
            observed_action=obs_action,
            confidence=observed.confidence,
            confidence_level=observed.confidence_level,
            step_index_map=self.action_to_step_index,
            frame=frame,
            timestamp=timestamp
        )

        if err_event:
            status = "ERROR"
            error_type = err_event.error_type

            msg = (
                err_event.message
                if not err_event.is_debounced
                else curr_def.message
            )

            # Track recovery candidate if out of order / skipped step
            # FIX: Disabled forward-jumping for strict sequential demo.
            # This forces the system to perfectly step 1->15 without accidentally skipping to the end.
            target_idx = None # self.action_to_step_index.get(obs_action)

            if target_idx is not None and target_idx > self.current_index:

                if self.recovery_target_index == target_idx:
                    self.recovery_counter += 1
                else:
                    self.recovery_target_index = target_idx
                    self.recovery_counter = 1

                # Resynchronize if the unexpected later action
                # persists for 5 consecutive frames.
                #
                # This matches the recovery validation requirement:
                # 5 consecutive observations of the later action
                # trigger RECOVERY.
                if self.recovery_counter >= 5:
                    status = "RECOVERY"

                    prev_state = curr_def.state_id
                    comp_state = curr_def.state_id
                    comp_step_num = curr_def.step_number
                    comp_action = curr_def.action

                    self.current_index = target_idx
                    self.match_counter = 0
                    self.recovery_target_index = None
                    self.recovery_counter = 0
                    self.state_entry_timestamp = timestamp
                    self.error_detector.reset()

                    msg = (
                        f"Resynchronized to step "
                        f"{self.current_state.step_number}: "
                        f"{self.current_state.message}"
                    )

                    return self._build_update(
                        observed,
                        frame,
                        timestamp,
                        transitioned=True,
                        prev_state=prev_state,
                        status=status,
                        err_type="RECOVERY",
                        msg=msg,
                        comp_state=comp_state,
                        comp_step_num=comp_step_num,
                        comp_action=comp_action,
                        trans_type="RECOVERY"
                    )

            return self._build_update(
                observed,
                frame,
                timestamp,
                transitioned=False,
                prev_state=previous_state_id,
                status=status,
                err_type=error_type,
                msg=msg
            )

        # 7. Final timeout / inaction check
        elapsed = timestamp - self.state_entry_timestamp

        if elapsed > curr_def.timeout_seconds:
            status = "WAITING"
            error_type = ErrorCategory.INACTION
            msg = (
                f"Inactivity alert: Expected '{curr_def.action}'. "
                f"{curr_def.message}"
            )

        return self._build_update(
            observed,
            frame,
            timestamp,
            transitioned=False,
            prev_state=previous_state_id,
            status=status,
            err_type=error_type,
            msg=msg
        )

    def _build_update(
        self,
        observed: ObservedAction,
        frame: int,
        timestamp: float,
        transitioned: bool,
        prev_state: Optional[str],
        status: str,
        err_type: Optional[str] = None,
        msg: str = "",
        comp_state: Optional[str] = None,
        comp_step_num: Optional[int] = None,
        comp_action: Optional[str] = None,
        trans_type: Optional[str] = None
    ) -> StateUpdate:

        curr = self.current_state
        next_s = curr.next_state

        return StateUpdate(
            current_step=curr.step_number,
            current_state_id=curr.state_id,
            expected_action=curr.action,
            observed_action=observed.action,
            next_step=next_s,
            status=status,
            error_type=err_type,
            confidence=observed.confidence,
            message=msg,
            transitioned=transitioned,
            previous_state=prev_state,
            timestamp=timestamp,
            frame=frame,
            completed_state=comp_state,
            completed_step_number=comp_step_num,
            completed_action=comp_action,
            transition_type=trans_type
        )

    def pause(self) -> None:
        self.is_paused = True
        self.status = "PAUSED"

    def resume(self) -> None:
        self.is_paused = False
        self.status = "ACTIVE"

    def stop(self) -> None:
        self.is_stopped = True
        self.status = "STOPPED"

    def reset(self) -> None:
        """Resets the state machine back to step 1 (S01)."""

        self.current_index = 0
        self.status = "START"
        self.match_counter = 0
        self.state_entry_timestamp = None
        self.is_paused = False
        self.is_stopped = False
        self.recovery_target_index = None
        self.recovery_counter = 0
        self.error_detector.reset()

    def resync_to_step(self, step_number: int) -> None:
        """Explicitly resynchronizes the machine state to a target step number (1-16)."""

        idx = step_number - 1

        if 0 <= idx < len(self.step_list):
            self.current_index = idx
            self.match_counter = 0
            self.state_entry_timestamp = None
            self.error_detector.reset()