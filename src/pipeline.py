"""
Master Experiment Pipeline
Coordinates normalized perception parsing, temporal history, action inference, state machine execution,
event logging, voice alerts, and UI callbacks.
"""
from typing import Dict, Any, Optional, Callable, Union, Tuple

from src.perception_types import NormalizedFrame
from src.input_adapter import Person1Adapter
from src.temporal_logic import TemporalHistoryBuffer
from src.action_inference import ActionInferenceEngine, ObservedAction
from src.state_machine import ExperimentStateMachine, StateUpdate
from src.event_logger import EventLogger
from src.voice_alert import VoiceAlertManager


class ExperimentPipeline:
    """
    Unified execution pipeline used by both Replay Mode and Live Camera Mode.
    """

    def __init__(
        self,
        config_path: str = "config/thresholds.json",
        sequence_config_path: str = "config/experiment_sequence.json",
        log_dir: str = "logs",
        use_mock_tts: bool = False,
        on_update_callback: Optional[Callable[[StateUpdate, NormalizedFrame, ObservedAction], None]] = None
    ):
        self.config_path = config_path
        self.sequence_config_path = sequence_config_path
        self.on_update_callback = on_update_callback

        self.buffer = TemporalHistoryBuffer(config_path=self.config_path)
        self.inference_engine = ActionInferenceEngine(buffer=self.buffer, config_path=self.config_path)
        self.state_machine = ExperimentStateMachine(sequence_config_path=self.sequence_config_path)

        self.logger = EventLogger(log_dir=log_dir, config_path=self.config_path)
        self.voice_manager = VoiceAlertManager(config_path=self.config_path, use_mock_tts=use_mock_tts)

    def process_frame(self, raw_data: Union[str, Dict[str, Any]]) -> Tuple[StateUpdate, NormalizedFrame, ObservedAction]:
        """
        Processes a single raw frame record through the complete downstream pipeline.
        """
        # 1. Parse into NormalizedFrame
        norm_frame: NormalizedFrame = Person1Adapter.parse_record(raw_data)

        # 2. Action Inference (updates TemporalHistoryBuffer internally)
        # Pass current step number so OPEN vs CLOSE white box is resolved by context
        observed_action: ObservedAction = self.inference_engine.process_frame(
            norm_frame,
            current_step_number=self.state_machine.current_state.step_number
        )

        # 3. State Machine Update
        state_update: StateUpdate = self.state_machine.update(
            observed=observed_action,
            frame=norm_frame.frame,
            timestamp=norm_frame.timestamp
        )

        # 4. Structured Event Logger
        self.logger.log_state_update(state_update)

        # 5. Offline Voice Alerts
        self.voice_manager.speak_state_update(state_update)

        # 6. UI Callback Trigger
        if self.on_update_callback:
            self.on_update_callback(state_update, norm_frame, observed_action)

        return state_update, norm_frame, observed_action

    def start_experiment_announcement(self) -> None:
        """Speaks 'Let's start the experiment!' and guides person to Step 1."""
        self.voice_manager.speak_experiment_start()

    def reset(self) -> None:
        """Resets all pipeline components."""
        self.buffer.clear()
        self.state_machine.reset()
        self.voice_manager.reset()

    def close(self) -> None:
        """Flushes and closes underlying loggers."""
        self.logger.close()
