"""
Replay Controller & Worker Thread
Executes sequential dataset playback simulating a live perception stream with replay speed control.
"""
import os
import time
import threading
from typing import Optional, Callable, Dict, Any, List, Tuple

from src.pipeline import ExperimentPipeline
from src.state_machine import StateUpdate
from src.perception_types import NormalizedFrame
from src.action_inference import ObservedAction


class ReplayController:
    """
    Asynchronous replay engine streaming frames sequentially into the ExperimentPipeline.
    """

    ALLOWED_SPEEDS = [0.25, 0.5, 1.0, 2.0, 4.0]

    def __init__(
        self,
        dataset_path: str = "data/person1_visual_output.jsonl",
        pipeline: Optional[ExperimentPipeline] = None,
        replay_speed: float = 1.0,
        on_update_callback: Optional[Callable[[StateUpdate, NormalizedFrame, ObservedAction], None]] = None,
        on_complete_callback: Optional[Callable[[], None]] = None
    ):
        self.dataset_path = dataset_path
        self.replay_speed = replay_speed if replay_speed in self.ALLOWED_SPEEDS else 1.0
        self.on_update_callback = on_update_callback
        self.on_complete_callback = on_complete_callback

        self.pipeline = pipeline if pipeline is not None else ExperimentPipeline(on_update_callback=on_update_callback)

        self.is_running: bool = False
        self.is_paused: bool = False
        self.is_stopped: bool = False
        self.is_completed: bool = False

        self._thread: Optional[threading.Thread] = None
        self._pause_event = threading.Event()
        self._pause_event.set()  # set = unpaused

        # Statistics tracking
        self.frames_processed: int = 0
        self.start_wall_time: Optional[float] = None
        self.end_wall_time: Optional[float] = None
        self.observed_action_counts: Dict[str, int] = {}
        self.state_transitions: List[Tuple[int, float, str, str]] = []  # (frame, ts, prev_state, new_state)
        self.errors_detected: List[Tuple[int, float, str, str]] = []  # (frame, ts, err_type, msg)
        self.uncertain_frames_count: int = 0

    def start(self) -> None:
        """Starts the replay loop in a background thread."""
        if self.is_running:
            return

        self.is_running = True
        self.is_paused = False
        self.is_stopped = False
        self.is_completed = False
        self._pause_event.set()

        self._thread = threading.Thread(target=self._replay_loop, daemon=True)
        self._thread.start()

    def pause(self) -> None:
        """Pauses the replay loop."""
        self.is_paused = True
        self._pause_event.clear()
        self.pipeline.state_machine.pause()

    def resume(self) -> None:
        """Resumes a paused replay loop."""
        self.is_paused = False
        self._pause_event.set()
        self.pipeline.state_machine.resume()

    def stop(self) -> None:
        """Stops the replay loop."""
        self.is_stopped = True
        self.is_running = False
        self._pause_event.set()
        self.pipeline.state_machine.stop()

    def reset(self) -> None:
        """Stops and resets the replay engine and pipeline."""
        self.stop()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        self.frames_processed = 0
        self.start_wall_time = None
        self.end_wall_time = None
        self.observed_action_counts.clear()
        self.state_transitions.clear()
        self.errors_detected.clear()
        self.uncertain_frames_count = 0
        self.is_completed = False

        self.pipeline.reset()

    def set_speed(self, speed: float) -> None:
        """Sets the replay speed multiplier (0.25x, 0.5x, 1x, 2x, 4x)."""
        if speed in self.ALLOWED_SPEEDS:
            self.replay_speed = speed

    def _replay_loop(self) -> None:
        if not os.path.exists(self.dataset_path):
            print(f"Error: Dataset file '{self.dataset_path}' not found.")
            self.is_running = False
            return

        self.start_wall_time = time.time()
        prev_dataset_ts: Optional[float] = None

        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if self.is_stopped:
                    break

                self._pause_event.wait()  # Block if paused

                if self.is_stopped:
                    break

                line_str = line.strip()
                if not line_str:
                    continue

                # Process frame
                update, frame_obj, obs_action = self.pipeline.process_frame(line_str)
                self.frames_processed += 1

                # Update stats
                act_name = obs_action.action
                self.observed_action_counts[act_name] = self.observed_action_counts.get(act_name, 0) + 1

                if obs_action.confidence_level == "UNCERTAIN" or act_name == "UNCERTAIN":
                    self.uncertain_frames_count += 1

                if update.transitioned:
                    self.state_transitions.append(
                        (frame_obj.frame, frame_obj.timestamp, update.previous_state or "", update.current_state_id)
                    )

                if update.status == "ERROR" and update.error_type:
                    self.errors_detected.append(
                        (frame_obj.frame, frame_obj.timestamp, update.error_type, update.message)
                    )

                # Simulate live streaming delay based on dataset timestamp difference and replay speed
                if prev_dataset_ts is not None:
                    delta_ts = max(0.0, frame_obj.timestamp - prev_dataset_ts)
                    sleep_duration = delta_ts / self.replay_speed
                    if sleep_duration > 0.001:
                        time.sleep(sleep_duration)

                prev_dataset_ts = frame_obj.timestamp

        self.end_wall_time = time.time()
        self.is_running = False
        self.is_completed = True

        if self.on_complete_callback and not self.is_stopped:
            self.on_complete_callback()
