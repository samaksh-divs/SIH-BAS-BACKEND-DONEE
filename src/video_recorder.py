"""
Local Video Recorder Module (Phase 11)
Asynchronously records live camera video frames to local disk using OpenCV cv2.VideoWriter.
Outputs unique timestamped files under recordings/experiment_YYYYMMDD_HHMMSS.mp4.
Runs frame writing in a background thread to prevent disk I/O from stalling perception.
"""
import os
import time
import queue
import threading
from typing import Optional, Any

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class VideoRecorder:
    """
    Asynchronous local MP4 video recorder for experiment runs.
    """

    def __init__(self, output_dir: str = "recordings", fps: float = 5.0):
        self.output_dir = output_dir
        self.fps = fps

        self.is_recording: bool = False
        self.recording_path: Optional[str] = None
        self.writer = None

        self._queue = queue.Queue(maxsize=300)
        self._thread: Optional[threading.Thread] = None

    def start_recording(self, width: int = 640, height: int = 480) -> Optional[str]:
        """
        Starts video recording and initializes OpenCV VideoWriter.
        Returns the output filename path.
        """
        if self.is_recording:
            return self.recording_path

        try:
            os.makedirs(self.output_dir, exist_ok=True)
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            self.recording_path = os.path.join(self.output_dir, f"experiment_{timestamp_str}.mp4")

            if HAS_CV2:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.writer = cv2.VideoWriter(self.recording_path, fourcc, self.fps, (width, height))

            self.is_recording = True
            self._queue = queue.Queue(maxsize=300)
            self._thread = threading.Thread(target=self._writer_loop, daemon=True)
            self._thread.start()
            return self.recording_path
        except Exception:
            self.is_recording = False
            self.recording_path = None
            self.writer = None
            return None

    def write_frame(self, frame_data: Any) -> bool:
        """
        Enqueues a video frame for asynchronous writing.
        """
        if not self.is_recording or frame_data is None:
            return False

        try:
            self._queue.put_nowait(frame_data)
            return True
        except queue.Full:
            return False

    def _writer_loop(self) -> None:
        """Background worker thread draining frame queue and writing to disk."""
        while self.is_recording or not self._queue.empty():
            try:
                frame = self._queue.get(timeout=0.2)
                if HAS_CV2 and self.writer is not None:
                    self.writer.write(frame)
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                pass

    def stop_recording(self) -> Optional[str]:
        """
        Stops recording, flushes queue, releases VideoWriter, and finalizes output file.
        """
        if not self.is_recording:
            return self.recording_path

        self.is_recording = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            self._thread = None

        if HAS_CV2 and self.writer is not None:
            try:
                self.writer.release()
            except Exception:
                pass
            self.writer = None

        saved_path = self.recording_path
        return saved_path

    def close(self) -> None:
        """Alias for stop_recording."""
        self.stop_recording()
