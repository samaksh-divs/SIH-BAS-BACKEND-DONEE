"""
Structured Event Logger
Generates lightweight timestamped JSON Lines (.jsonl) event logs with explicit step completion semantics and debouncing.
"""
import json
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional

from src.state_machine import StateUpdate


class EventLogger:
    """
    Structured event logger writing JSON Lines records to disk.
    Filters out frame-by-frame duplicate events using debouncing rules.
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        config_path: Optional[str] = "config/thresholds.json",
        custom_filename: Optional[str] = None
    ):
        self.cooldown_seconds = 10.0
        self.log_dir = log_dir if log_dir is not None else "logs"

        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f).get("logging", {})
                if log_dir is None:
                    self.log_dir = cfg.get("log_directory", self.log_dir)
                self.cooldown_seconds = float(cfg.get("duplicate_cooldown_seconds", self.cooldown_seconds))

        os.makedirs(self.log_dir, exist_ok=True)

        if custom_filename:
            self.filepath = os.path.join(self.log_dir, custom_filename)
        else:
            time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.filepath = os.path.join(self.log_dir, f"experiment_log_{time_str}.jsonl")

        self.file_handle = open(self.filepath, 'a', encoding='utf-8')

        # De-duplication tracking
        self.last_logged_key: Optional[str] = None
        self.last_logged_timestamp: float = -999.0
        self.logged_events_count: int = 0
        self.start_timestamp: Optional[float] = None
        self.end_timestamp: Optional[float] = None
        self.recording_path: Optional[str] = None

    def set_recording_path(self, recording_path: str) -> None:
        """Associates active video recording file path with event logs."""
        self.recording_path = recording_path

    def log_record(self, record: Dict[str, Any], dedup_key: str, timestamp: float, force: bool = False) -> Optional[Dict[str, Any]]:
        """Writes a pre-formatted dict record to disk if not debounced."""
        if self.recording_path:
            record["recording_file"] = self.recording_path

        time_since_last = timestamp - self.last_logged_timestamp

        if not force and dedup_key == self.last_logged_key and time_since_last < self.cooldown_seconds:
            return None

        if self.start_timestamp is None:
            self.start_timestamp = timestamp
        self.end_timestamp = timestamp

        if self.file_handle is None or getattr(self.file_handle, 'closed', False):
            return None

        line = json.dumps(record) + "\n"
        try:
            self.file_handle.write(line)
            self.file_handle.flush()
        except (ValueError, OSError):
            return None

        self.last_logged_key = dedup_key
        self.last_logged_timestamp = timestamp
        self.logged_events_count += 1

        return record

    def log_event(
        self,
        timestamp: float,
        frame: int,
        event: str,
        state: str,
        step_number: int,
        expected_action: str,
        observed_action: str,
        status: str,
        confidence: float = 1.0,
        message: str = "",
        force: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Writes a standard structured JSON event record.
        """
        dedup_key = f"{event}:{state}:{step_number}:{expected_action}:{observed_action}:{status}"
        record = {
            "timestamp": round(float(timestamp), 3),
            "frame": int(frame),
            "event": str(event),
            "state": str(state),
            "step_number": int(step_number),
            "expected_action": str(expected_action),
            "observed_action": str(observed_action),
            "status": str(status),
            "confidence": round(float(confidence), 2),
            "message": str(message)
        }
        return self.log_record(record, dedup_key, timestamp, force=force)

    def log_step_completed(
        self,
        timestamp: float,
        frame: int,
        completed_state: str,
        completed_step_number: int,
        completed_action: str,
        next_state: str,
        next_step_number: int,
        next_action: str,
        confidence: float = 1.0,
        message: str = "",
        force: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Writes a STEP_COMPLETED event record with unambiguous completed vs next step semantics.
        """
        dedup_key = f"STEP_COMPLETED:{completed_state}:{next_state}"
        record = {
            "timestamp": round(float(timestamp), 3),
            "frame": int(frame),
            "event": "STEP_COMPLETED",
            "completed_state": str(completed_state),
            "completed_step_number": int(completed_step_number),
            "completed_action": str(completed_action),
            "next_state": str(next_state),
            "next_step_number": int(next_step_number),
            "next_action": str(next_action),
            "status": "STEP_COMPLETED",
            "confidence": round(float(confidence), 2),
            "message": str(message)
        }
        return self.log_record(record, dedup_key, timestamp, force=force)

    def log_experiment_completed(
        self,
        timestamp: float,
        frame: int,
        completed_state: str = "S15",
        completed_step_number: int = 15,
        completed_action: str = "CLOSE_WHITE_BOX",
        confidence: float = 1.0,
        message: str = "Experiment complete.",
        force: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Writes an EXPERIMENT_COMPLETED event record.
        """
        dedup_key = "EXPERIMENT_COMPLETED"
        record = {
            "timestamp": round(float(timestamp), 3),
            "frame": int(frame),
            "event": "EXPERIMENT_COMPLETED",
            "completed_state": str(completed_state),
            "completed_step_number": int(completed_step_number),
            "completed_action": str(completed_action),
            "next_state": "S16",
            "next_step_number": 16,
            "next_action": "COMPLETE",
            "status": "COMPLETE",
            "confidence": round(float(confidence), 2),
            "message": str(message)
        }
        return self.log_record(record, dedup_key, timestamp, force=force)

    def log_state_update(self, update: StateUpdate, force: bool = False) -> Optional[Dict[str, Any]]:
        """
        Translates a StateUpdate into a structured log event with corrected transition semantics.
        """
        if update.status == "STEP_COMPLETED":
            comp_state = update.completed_state if update.completed_state else (update.previous_state or f"S{update.current_step-1:02d}")
            comp_step = update.completed_step_number if update.completed_step_number else max(1, update.current_step - 1)
            comp_act = update.completed_action if update.completed_action else update.observed_action

            return self.log_step_completed(
                timestamp=update.timestamp,
                frame=update.frame,
                completed_state=comp_state,
                completed_step_number=comp_step,
                completed_action=comp_act,
                next_state=update.current_state_id,
                next_step_number=update.current_step,
                next_action=update.expected_action,
                confidence=update.confidence,
                message=update.message,
                force=True
            )

        if update.status == "COMPLETE":
            comp_state = update.completed_state if update.completed_state else "S15"
            comp_step = update.completed_step_number if update.completed_step_number else 15
            comp_act = update.completed_action if update.completed_action else "CLOSE_WHITE_BOX"

            return self.log_experiment_completed(
                timestamp=update.timestamp,
                frame=update.frame,
                completed_state=comp_state,
                completed_step_number=comp_step,
                completed_action=comp_act,
                confidence=update.confidence,
                message=update.message,
                force=True
            )

        event_type = "STEP_IN_PROGRESS"
        if update.status == "START":
            event_type = "EXPERIMENT_STARTED"
        elif update.status == "ERROR":
            event_type = update.error_type if update.error_type else "STEP_ERROR"
        elif update.status == "RECOVERY":
            event_type = "RECOVERY"
            force = True
        elif update.status == "WAITING":
            event_type = update.error_type if update.error_type else "STEP_WAITING"
        elif update.status == "PAUSED":
            event_type = "EXPERIMENT_PAUSED"
            force = True
        elif update.status == "RESUMED":
            event_type = "EXPERIMENT_RESUMED"
            force = True

        return self.log_event(
            timestamp=update.timestamp,
            frame=update.frame,
            event=event_type,
            state=update.current_state_id,
            step_number=update.current_step,
            expected_action=update.expected_action,
            observed_action=update.observed_action,
            status=update.status,
            confidence=update.confidence,
            message=update.message,
            force=force
        )

    def close(self) -> None:
        """Closes the underlying log file handle safely."""
        if hasattr(self, 'file_handle') and self.file_handle and not self.file_handle.closed:
            if self.start_timestamp is not None and self.end_timestamp is not None:
                duration = round(self.end_timestamp - self.start_timestamp, 2)
                summary_record = {
                    "event": "LOG_CLOSED",
                    "total_events": self.logged_events_count,
                    "duration_seconds": duration
                }
                self.file_handle.write(json.dumps(summary_record) + "\n")
            self.file_handle.flush()
            self.file_handle.close()
