"""
Live Camera Manager Module
Handles OpenCV video capture, frame timing, telemetry (FPS, Avg/P95/Max Latency, Dropped Frames),
Person 1 inference dispatch, Local Video Recording (Phase 11), and IP Streaming (Phase 12).
NO SILENT FALLBACK: If physical camera fails to open, sets status to CAMERA_UNAVAILABLE and stops.
"""
import time
import threading
from typing import Optional, Callable, Dict, Any, List

from src.pipeline import ExperimentPipeline
from src.person1_live_adapter import Person1LiveAdapter
from src.video_recorder import VideoRecorder
from src.streaming import IPStreamer
from src.state_machine import StateUpdate
from src.perception_types import NormalizedFrame
from src.action_inference import ObservedAction

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class LiveCameraManager:
    """
    Asynchronous live camera manager with strict error reporting, metrics telemetry,
    local recording integration, and IP streaming fan-out.
    """

    def __init__(
        self,
        device_index: int = 0,
        width: int = 640,
        height: int = 480,
        target_fps: int = 30,
        pipeline: Optional[ExperimentPipeline] = None,
        adapter: Optional[Person1LiveAdapter] = None,
        recorder: Optional[VideoRecorder] = None,
        streamer: Optional[IPStreamer] = None,
        on_frame_callback: Optional[Callable[[Any, StateUpdate, NormalizedFrame, ObservedAction, Dict[str, Any]], None]] = None,
        mock_camera: bool = False
    ):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.on_frame_callback = on_frame_callback
        self.mock_camera = mock_camera

        self.pipeline = pipeline if pipeline is not None else ExperimentPipeline()
        self.adapter = adapter if adapter is not None else Person1LiveAdapter(mock_mode=self.mock_camera)
        self.recorder = recorder if recorder is not None else VideoRecorder()
        self.streamer = streamer if streamer is not None else IPStreamer()

        self.is_running: bool = False
        self.is_paused: bool = False
        self.is_stopped: bool = True
        self.is_experiment_started: bool = False
        self.camera_status: str = "STOPPED"  # STOPPED, CONNECTED, CAMERA_UNAVAILABLE, MOCK_ACTIVE

        self._thread: Optional[threading.Thread] = None
        self._cap = None

        # Performance Metrics Telemetry
        self.camera_fps: float = 0.0
        self.person1_fps: float = 0.0
        self.processing_fps: float = 0.0
        self.avg_latency_ms: float = 0.0
        self.p95_latency_ms: float = 0.0
        self.max_latency_ms: float = 0.0
        self.dropped_frames: int = 0
        self.frames_captured: int = 0

        self._frame_times: List[float] = []
        self._p1_durations: List[float] = []
        self._proc_durations: List[float] = []
        self._latencies_ms: List[float] = []

    def start_camera(self) -> bool:
        """
        Initializes camera hardware/stream and launches background capture thread.
        NO SILENT FALLBACK: Returns False and stays stopped if camera device fails to open.
        """
        if self.is_running:
            return True

        if self.mock_camera:
            self.camera_status = "MOCK_ACTIVE"
            self.is_stopped = False
            self.is_running = True
            self.is_paused = False
            self.adapter.mock_mode = True
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()
            return True

        if not HAS_CV2:
            self.camera_status = "CAMERA_UNAVAILABLE"
            self.is_running = False
            self.is_stopped = True
            return False

        try:
            self._cap = cv2.VideoCapture(self.device_index)
            if self._cap and self._cap.isOpened():
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.camera_status = "CONNECTED"
                self.is_stopped = False
                self.is_running = True
                self.is_paused = False
                self._thread = threading.Thread(target=self._capture_loop, daemon=True)
                self._thread.start()
                return True
            else:
                self.camera_status = "CAMERA_UNAVAILABLE"
                self.is_running = False
                self.is_stopped = True
                self._cap = None
                return False
        except Exception:
            self.camera_status = "CAMERA_UNAVAILABLE"
            self.is_running = False
            self.is_stopped = True
            self._cap = None
            return False

    def stop_camera(self) -> None:
        """Stops background capture thread, releases camera hardware, and stops recording/streaming."""
        self.is_running = False
        self.is_stopped = True
        self.camera_status = "STOPPED"

        if self._thread and self._thread.is_alive():
            try:
                self._thread.join(timeout=1.0)
            except Exception:
                pass
            self._thread = None

        if self._cap and HAS_CV2:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

        if self.recorder.is_recording:
            self.recorder.stop_recording()

        if self.streamer.is_streaming:
            self.streamer.stop_stream()

    def start_experiment(self) -> None:
        """Starts state machine execution, video recording, and event logging."""
        self.is_experiment_started = True
        self.pipeline.state_machine.reset()
        self.pipeline.state_machine.status = "ACTIVE"

        # Start Video Recorder
        rec_path = self.recorder.start_recording(width=self.width, height=self.height)
        if rec_path:
            self.pipeline.logger.set_recording_path(rec_path)

        # Start IP Streamer if enabled in config
        if self.streamer.config.get("stream_enabled", False):
            self.streamer.start_stream()

        if self.pipeline.logger.start_timestamp is None:
            self.pipeline.logger.log_event(
                timestamp=time.time(),
                frame=1,
                event="EXPERIMENT_STARTED",
                state="S01",
                step_number=1,
                expected_action="OPEN_WHITE_BOX",
                observed_action="OPEN_WHITE_BOX",
                status="START",
                confidence=1.0,
                message="Started live experiment tracking S01"
            )

    def pause_experiment(self) -> None:
        self.is_paused = True
        self.pipeline.state_machine.pause()

    def resume_experiment(self) -> None:
        self.is_paused = False
        self.pipeline.state_machine.resume()

    def reset_experiment(self) -> None:
        """Resets state machine, pipeline, metrics, recorder, and logger."""
        self.is_experiment_started = False
        self.frames_captured = 0
        self.dropped_frames = 0
        self.camera_fps = 0.0
        self.person1_fps = 0.0
        self.processing_fps = 0.0
        self.avg_latency_ms = 0.0
        self.p95_latency_ms = 0.0
        self.max_latency_ms = 0.0
        self._frame_times.clear()
        self._p1_durations.clear()
        self._proc_durations.clear()
        self._latencies_ms.clear()

        if self.recorder.is_recording:
            self.recorder.stop_recording()

        self.pipeline.reset()

    def set_device_index(self, index: int) -> None:
        was_running = self.is_running
        if was_running:
            self.stop_camera()
        self.device_index = index
        if was_running:
            self.start_camera()

    def _capture_loop(self) -> None:
        frame_num = 0

        while self.is_running and not self.is_stopped:
            loop_start = time.time()
            capture_ts = loop_start
            frame_num += 1

            cv_image = None
            if HAS_CV2 and self._cap and self._cap.isOpened():
                ret, frame_data = self._cap.read()
                if ret:
                    cv_image = frame_data
                else:
                    self.dropped_frames += 1
                    # Generate frame matrix for model execution fallback
                    import numpy as np
                    cv_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                    time.sleep(1.0 / self.target_fps)
            else:
                import numpy as np
                cv_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                time.sleep(1.0 / self.target_fps)

            # Measure Capture FPS
            self.frames_captured += 1
            now = time.time()
            self._frame_times.append(now)
            if len(self._frame_times) > 30:
                self._frame_times.pop(0)

            if len(self._frame_times) > 1:
                fps_span = self._frame_times[-1] - self._frame_times[0]
                if fps_span > 0:
                    self.camera_fps = round((len(self._frame_times) - 1) / fps_span, 1)

            # 1. Asynchronous Fan-Out to Local Video Recorder
            if self.recorder.is_recording and cv_image is not None:
                self.recorder.write_frame(cv_image)

            # 2. Asynchronous Fan-Out to IP Video Streamer
            if self.streamer.is_streaming and cv_image is not None:
                self.streamer.push_frame(cv_image)

            # 3. Person 1 Inference
            p1_start = time.time()
            perception_record = self.adapter.process_live_frame(
                frame_num=frame_num,
                timestamp=round(capture_ts, 3),
                image_matrix=cv_image
            )
            p1_end = time.time()
            p1_dur = p1_end - p1_start
            self._p1_durations.append(p1_dur)
            if len(self._p1_durations) > 30:
                self._p1_durations.pop(0)
            avg_p1 = sum(self._p1_durations) / len(self._p1_durations)
            if avg_p1 > 0:
                self.person1_fps = round(1.0 / avg_p1, 1)

            # 4. Downstream Person 2 ExperimentPipeline
            p_start = time.time()
            update, norm_frame, obs_action = self.pipeline.process_frame(perception_record)
            p_end = time.time()
            proc_dur = p_end - p_start

            self._proc_durations.append(proc_dur)
            if len(self._proc_durations) > 30:
                self._proc_durations.pop(0)

            avg_proc = sum(self._proc_durations) / len(self._proc_durations)
            if avg_proc > 0:
                self.processing_fps = round(1.0 / avg_proc, 1)

            # Latency Metrics (Turnaround from capture timestamp)
            lat_ms = (p_end - capture_ts) * 1000.0
            self._latencies_ms.append(lat_ms)
            if len(self._latencies_ms) > 100:
                self._latencies_ms.pop(0)

            self.avg_latency_ms = round(sum(self._latencies_ms) / len(self._latencies_ms), 1)
            sorted_lat = sorted(self._latencies_ms)
            p95_idx = int(len(sorted_lat) * 0.95)
            self.p95_latency_ms = round(sorted_lat[min(p95_idx, len(sorted_lat) - 1)], 1)
            self.max_latency_ms = round(sorted_lat[-1], 1)

            metrics = {
                "camera_fps": self.camera_fps,
                "person1_fps": self.person1_fps,
                "processing_fps": self.processing_fps,
                "avg_latency_ms": self.avg_latency_ms,
                "p95_latency_ms": self.p95_latency_ms,
                "max_latency_ms": self.max_latency_ms,
                "dropped_frames": self.dropped_frames,
                "frames_captured": self.frames_captured,
                "mock_camera": self.mock_camera,
                "camera_status": self.camera_status,
                "p1_status": self.adapter.adapter_status,
                "recording_active": self.recorder.is_recording,
                "recording_path": self.recorder.recording_path,
                "stream_status": self.streamer.status,
                "stream_url": self.streamer.get_stream_url() if self.streamer.is_streaming else "OFF"
            }

            if self.on_frame_callback:
                self.on_frame_callback(cv_image, update, norm_frame, obs_action, metrics)
