import sys
import os
import time
import threading
import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.live_camera import LiveCameraManager
from src.person1_live_adapter import Person1LiveAdapter
from src.video_recorder import VideoRecorder
from src.streaming import IPStreamer
from src.pipeline import ExperimentPipeline


class LiveDemoGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SIH BAS - Live AI Detection Demo")
        self.root.geometry("1400x850")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.running = False
        self.latest_frame = None
        self.latest_update = None
        self.latest_norm_frame = None
        self.latest_action = None
        self.latest_metrics = {}

        self.frame_lock = threading.Lock()

        # ---------------------------------------------------------
        # BACKEND
        # ---------------------------------------------------------
        print("Loading Person 1 AI...")

        self.adapter = Person1LiveAdapter(
            config_path="config/person1_live.json",
            mock_mode=False
        )

        self.recorder = VideoRecorder(output_dir="recordings")
        self.streamer = IPStreamer(
            config_path="config/streaming.json"
        )
        self.pipeline = ExperimentPipeline(
            use_mock_tts=True
        )

        self.cam_mgr = LiveCameraManager(
            device_index=0,
            pipeline=self.pipeline,
            adapter=self.adapter,
            recorder=self.recorder,
            streamer=self.streamer,
            mock_camera=False
        )

        self.cam_mgr.on_frame_callback = self.frame_callback

        self.build_ui()

        self.root.after(30, self.update_gui)

    # ============================================================
    # UI
    # ============================================================

    def build_ui(self):

        # Header
        header = tk.Frame(
            self.root,
            bg="#172033",
            height=65
        )
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(
            header,
            text="SIH BAS — LIVE AI PROCEDURE MONITOR",
            fg="white",
            bg="#172033",
            font=("Segoe UI", 20, "bold")
        ).pack(side="left", padx=20)

        self.connection_label = tk.Label(
            header,
            text="● INITIALIZING",
            fg="#f0ad4e",
            bg="#172033",
            font=("Segoe UI", 12, "bold")
        )
        self.connection_label.pack(side="right", padx=20)

        # Main area
        main = tk.Frame(self.root, bg="#eef1f5")
        main.pack(fill="both", expand=True)

        # ---------------------------------------------------------
        # LEFT: CAMERA
        # ---------------------------------------------------------

        left = tk.Frame(main, bg="#111827")
        left.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(12, 6),
            pady=12
        )

        tk.Label(
            left,
            text="LIVE CAMERA",
            fg="white",
            bg="#111827",
            font=("Segoe UI", 13, "bold")
        ).pack(anchor="w", padx=12, pady=(10, 5))

        self.video_label = tk.Label(
            left,
            bg="black"
        )
        self.video_label.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10
        )

        # ---------------------------------------------------------
        # RIGHT PANEL
        # ---------------------------------------------------------

        right = tk.Frame(
            main,
            bg="#eef1f5",
            width=430
        )
        right.pack(
            side="right",
            fill="y",
            padx=(6, 12),
            pady=12
        )
        right.pack_propagate(False)

        # Current state
        state_card = tk.LabelFrame(
            right,
            text=" PROCEDURE STATE ",
            font=("Segoe UI", 11, "bold"),
            bg="white",
            padx=12,
            pady=10
        )
        state_card.pack(fill="x", pady=(0, 10))

        self.state_label = tk.Label(
            state_card,
            text="S01",
            font=("Segoe UI", 28, "bold"),
            bg="white"
        )
        self.state_label.pack()

        self.expected_label = tk.Label(
            state_card,
            text="Expected: OPEN_WHITE_BOX",
            font=("Segoe UI", 11),
            bg="white",
            wraplength=360
        )
        self.expected_label.pack(pady=5)

        self.status_label = tk.Label(
            state_card,
            text="WAITING",
            font=("Segoe UI", 11, "bold"),
            bg="white"
        )
        self.status_label.pack()

        # Action
        action_card = tk.LabelFrame(
            right,
            text=" ACTION INFERENCE ",
            font=("Segoe UI", 11, "bold"),
            bg="white",
            padx=12,
            pady=10
        )
        action_card.pack(fill="x", pady=(0, 10))

        self.action_label = tk.Label(
            action_card,
            text="UNCERTAIN",
            font=("Segoe UI", 16, "bold"),
            bg="white",
            wraplength=360
        )
        self.action_label.pack()

        self.confidence_label = tk.Label(
            action_card,
            text="Confidence: 0.00",
            font=("Segoe UI", 10),
            bg="white"
        )
        self.confidence_label.pack(pady=5)

        # Objects
        object_card = tk.LabelFrame(
            right,
            text=" DETECTED OBJECTS ",
            font=("Segoe UI", 11, "bold"),
            bg="white",
            padx=12,
            pady=8
        )
        object_card.pack(fill="x", pady=(0, 10))

        self.objects_text = tk.Label(
            object_card,
            text="No objects detected",
            font=("Consolas", 10),
            bg="white",
            justify="left",
            anchor="w",
            wraplength=360
        )
        self.objects_text.pack(fill="x")

        # Interaction signals
        signal_card = tk.LabelFrame(
            right,
            text=" HAND / OBJECT INTERACTION ",
            font=("Segoe UI", 11, "bold"),
            bg="white",
            padx=12,
            pady=8
        )
        signal_card.pack(fill="x", pady=(0, 10))

        self.signals_text = tk.Label(
            signal_card,
            text="No interaction signals",
            font=("Consolas", 9),
            bg="white",
            justify="left",
            anchor="w",
            wraplength=360
        )
        self.signals_text.pack(fill="x")

        # Telemetry
        telemetry_card = tk.LabelFrame(
            right,
            text=" LIVE TELEMETRY ",
            font=("Segoe UI", 11, "bold"),
            bg="white",
            padx=12,
            pady=8
        )
        telemetry_card.pack(fill="x", pady=(0, 10))

        self.telemetry_text = tk.Label(
            telemetry_card,
            text="Camera FPS: --\n"
                 "Person 1 FPS: --\n"
                 "Processing FPS: --\n"
                 "Latency: -- ms\n"
                 "Dropped Frames: --",
            font=("Consolas", 10),
            bg="white",
            justify="left",
            anchor="w"
        )
        self.telemetry_text.pack(fill="x")

        # Controls
        controls = tk.Frame(
            right,
            bg="#eef1f5"
        )
        controls.pack(fill="x", pady=5)

        self.start_button = ttk.Button(
            controls,
            text="START",
            command=self.start
        )
        self.start_button.pack(
            side="left",
            expand=True,
            fill="x",
            padx=3
        )

        self.pause_button = ttk.Button(
            controls,
            text="PAUSE",
            command=self.pause
        )
        self.pause_button.pack(
            side="left",
            expand=True,
            fill="x",
            padx=3
        )

        self.resume_button = ttk.Button(
            controls,
            text="RESUME",
            command=self.resume
        )
        self.resume_button.pack(
            side="left",
            expand=True,
            fill="x",
            padx=3
        )

        self.stop_button = ttk.Button(
            controls,
            text="STOP",
            command=self.stop
        )
        self.stop_button.pack(
            side="left",
            expand=True,
            fill="x",
            padx=3
        )

    # ============================================================
    # CALLBACK FROM BACKEND
    # ============================================================

    def frame_callback(
        self,
        cv_image,
        update,
        norm_frame,
        obs_action,
        metrics
    ):
        with self.frame_lock:

            self.latest_frame = cv_image.copy()
            self.latest_update = update
            self.latest_norm_frame = norm_frame
            self.latest_action = obs_action
            self.latest_metrics = dict(metrics)

    # ============================================================
    # DRAW CAMERA
    # ============================================================

    def draw_frame(self, frame, norm_frame):

        if frame is None:
            return None

        display = frame.copy()

        # Try to draw normalized objects.
        if norm_frame is not None:

            for obj in getattr(norm_frame, "objects", []):

                name = getattr(
                    obj,
                    "class_name",
                    getattr(obj, "class", "object")
                )

                confidence = getattr(
                    obj,
                    "confidence",
                    getattr(obj, "score", None)
                )

                bbox = None

                for attr in [
                    "bbox",
                    "box",
                    "xyxy",
                    "bounding_box"
                ]:
                    if hasattr(obj, attr):
                        bbox = getattr(obj, attr)
                        break

                if bbox is not None:

                    try:
                        x1, y1, x2, y2 = map(
                            int,
                            bbox
                        )

                        cv2.rectangle(
                            display,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),
                            2
                        )

                        label = str(name)

                        if confidence is not None:
                            try:
                                label += f" {float(confidence):.2f}"
                            except Exception:
                                pass

                        cv2.putText(
                            display,
                            label,
                            (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 255, 0),
                            2
                        )

                    except Exception:
                        pass

        # Overlay backend state directly on the video.
        if self.latest_update is not None:

            state = getattr(
                self.latest_update,
                "current_state_id",
                "S01"
            )

            status = getattr(
                self.latest_update,
                "status",
                "WAITING"
            )

            cv2.rectangle(
                display,
                (10, 10),
                (430, 85),
                (20, 20, 20),
                -1
            )

            cv2.putText(
                display,
                f"STATE: {state}",
                (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display,
                f"STATUS: {status}",
                (20, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

        return display

    # ============================================================
    # GUI UPDATE
    # ============================================================

    def update_gui(self):

        with self.frame_lock:

            frame = (
                self.latest_frame.copy()
                if self.latest_frame is not None
                else None
            )

            update = self.latest_update
            norm_frame = self.latest_norm_frame
            action = self.latest_action
            metrics = dict(self.latest_metrics)

        if frame is not None:

            frame = self.draw_frame(
                frame,
                norm_frame
            )

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            h, w = frame.shape[:2]

            max_w = 900
            max_h = 650

            scale = min(
                max_w / w,
                max_h / h,
                1.0
            )

            if scale < 1:
                frame = cv2.resize(
                    frame,
                    (
                        int(w * scale),
                        int(h * scale)
                    )
                )

            image = Image.fromarray(frame)

            photo = ImageTk.PhotoImage(
                image=image
            )

            self.video_label.configure(
                image=photo
            )

            self.video_label.image = photo

        if update is not None:

            state = getattr(
                update,
                "current_state_id",
                "S01"
            )

            expected = getattr(
                update,
                "expected_action",
                "OPEN_WHITE_BOX"
            )

            status = getattr(
                update,
                "status",
                "WAITING"
            )

            self.state_label.configure(
                text=state
            )

            self.expected_label.configure(
                text=f"Expected: {expected}"
            )

            self.status_label.configure(
                text=status
            )

        if action is not None:

            action_name = getattr(
                action,
                "action",
                "UNCERTAIN"
            )

            confidence = getattr(
                action,
                "confidence",
                0.0
            )

            self.action_label.configure(
                text=str(action_name)
            )

            try:
                self.confidence_label.configure(
                    text=f"Confidence: {float(confidence):.2f}"
                )
            except Exception:
                pass

        if norm_frame is not None:

            objects = getattr(
                norm_frame,
                "objects",
                []
            )

            names = []

            for obj in objects:

                name = getattr(
                    obj,
                    "class_name",
                    getattr(obj, "class", "unknown")
                )

                if name not in names:
                    names.append(str(name))

            if names:
                self.objects_text.configure(
                    text="\n".join(
                        "• " + x
                        for x in names
                    )
                )
            else:
                self.objects_text.configure(
                    text="No objects detected"
                )

            signals = getattr(
                norm_frame,
                "interaction_signals",
                {}
            )

            active_signals = [
                key
                for key, value in signals.items()
                if value
            ]

            if active_signals:

                self.signals_text.configure(
                    text="\n".join(
                        "• " + x
                        for x in active_signals
                    )
                )

            else:

                self.signals_text.configure(
                    text="No active hand/object signals"
                )

        # Telemetry
        camera_fps = getattr(
            self.cam_mgr,
            "camera_fps",
            0
        )

        p1_fps = getattr(
            self.cam_mgr,
            "person1_fps",
            0
        )

        proc_fps = getattr(
            self.cam_mgr,
            "processing_fps",
            0
        )

        latency = getattr(
            self.cam_mgr,
            "avg_latency_ms",
            0
        )

        dropped = getattr(
            self.cam_mgr,
            "dropped_frames",
            0
        )

        self.telemetry_text.configure(
            text=
                f"Camera FPS:       {camera_fps:.1f}\n"
                f"Person 1 FPS:     {p1_fps:.1f}\n"
                f"Processing FPS:   {proc_fps:.1f}\n"
                f"Avg Latency:      {latency:.1f} ms\n"
                f"Dropped Frames:   {dropped}"
        )

        # Connection status
        camera_status = str(
            getattr(
                self.cam_mgr,
                "camera_status",
                ""
            )
        )

        if camera_status.upper() == "CONNECTED":

            self.connection_label.configure(
                text="● CAMERA CONNECTED",
                fg="#42d392"
            )

        else:

            self.connection_label.configure(
                text=f"● {camera_status}",
                fg="#f0ad4e"
            )

        self.root.after(
            30,
            self.update_gui
        )

    # ============================================================
    # CONTROLS
    # ============================================================

    def start(self):

        if not self.running:

            try:
                self.cam_mgr.start_camera()
                time.sleep(0.5)

                self.cam_mgr.start_experiment()

                self.running = True

                self.connection_label.configure(
                    text="● RUNNING",
                    fg="#42d392"
                )

            except Exception as e:

                print(
                    "START ERROR:",
                    repr(e)
                )

    def pause(self):

        try:
            self.cam_mgr.pause_experiment()

            self.connection_label.configure(
                text="● PAUSED",
                fg="#f0ad4e"
            )

        except Exception as e:
            print(
                "PAUSE ERROR:",
                repr(e)
            )

    def resume(self):

        try:
            self.cam_mgr.resume_experiment()

            self.connection_label.configure(
                text="● RUNNING",
                fg="#42d392"
            )

        except Exception as e:
            print(
                "RESUME ERROR:",
                repr(e)
            )

    def stop(self):

        try:
            self.cam_mgr.stop_camera()

            self.running = False

            self.connection_label.configure(
                text="● STOPPED",
                fg="#ff6b6b"
            )

        except Exception as e:
            print(
                "STOP ERROR:",
                repr(e)
            )

    def close(self):

        try:
            self.cam_mgr.stop_camera()
        except Exception:
            pass

        self.root.destroy()


def main():

    root = tk.Tk()

    app = LiveDemoGUI(root)

    root.mainloop()


if __name__ == "__main__":
    main()
