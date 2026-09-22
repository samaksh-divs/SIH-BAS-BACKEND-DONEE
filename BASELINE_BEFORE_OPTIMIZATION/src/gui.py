"""
Monitoring GUI Module
Tkinter interface displaying real-time experiment status, 15-step progress, system health,
recent event logs, video area, real-time metrics, live camera controls, local recording status, and IP streaming status.
"""
import os
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional, Dict, Any, List

from src.pipeline import ExperimentPipeline
from src.replay import ReplayController
from src.live_camera import LiveCameraManager
from src.person1_live_adapter import Person1LiveAdapter
from src.video_recorder import VideoRecorder
from src.streaming import IPStreamer
from src.state_machine import StateUpdate
from src.perception_types import NormalizedFrame
from src.action_inference import ObservedAction


class ExperimentGUI:
    """
    Tkinter Graphical User Interface for monitoring autonomous experiment execution.
    """

    def __init__(self, root: tk.Tk, default_dataset_path: str = "data/person1_visual_output.jsonl", use_mock_tts: bool = False):
        self.root = root
        self.root.title("SIH EXPERIMENT 3 – Person 2 Execution Monitor")
        self.root.geometry("1280x900")
        self.root.minsize(1080, 750)

        self.update_queue = queue.Queue()

        # Initialize Pipeline
        self.pipeline = ExperimentPipeline(
            use_mock_tts=use_mock_tts,
            on_update_callback=self._queue_update_callback
        )

        # Initialize Replay Controller
        self.replay = ReplayController(
            dataset_path=default_dataset_path,
            pipeline=self.pipeline,
            on_complete_callback=self._queue_complete_callback
        )

        # Initialize Person 1 Adapter, Video Recorder & IP Streamer
        self.adapter = Person1LiveAdapter(mock_mode=False)
        self.recorder = VideoRecorder()
        self.streamer = IPStreamer()

        # Initialize Live Camera Manager
        self.live_camera = LiveCameraManager(
            device_index=0,
            pipeline=self.pipeline,
            adapter=self.adapter,
            recorder=self.recorder,
            streamer=self.streamer,
            on_frame_callback=self._queue_live_frame_callback
        )

        self._init_styles()
        self._build_ui()
        self._start_queue_listener()

    def _init_styles(self) -> None:
        style = ttk.Style()
        style.theme_use('clam')

        style.configure("Header.TLabel", font=("Helvetica", 18, "bold"), foreground="#1e293b")
        style.configure("CardTitle.TLabel", font=("Helvetica", 10, "bold"), foreground="#64748b")
        style.configure("CardVal.TLabel", font=("Helvetica", 14, "bold"), foreground="#0f172a")
        style.configure("Alert.TLabel", font=("Helvetica", 11, "bold"), foreground="#dc2626")
        style.configure("StatusGood.TLabel", font=("Helvetica", 12, "bold"), foreground="#16a34a")
        style.configure("StatusWarn.TLabel", font=("Helvetica", 12, "bold"), foreground="#ca8a04")
        style.configure("StatusErr.TLabel", font=("Helvetica", 12, "bold"), foreground="#dc2626")
        style.configure("Metric.TLabel", font=("Helvetica", 10, "bold"), foreground="#2563eb")

    def _build_ui(self) -> None:
        # Top Header Frame
        header_frame = ttk.Frame(self.root, padding=10)
        header_frame.pack(fill=tk.X)

        title_lbl = ttk.Label(header_frame, text="SIH EXPERIMENT 3 – AUTONOMOUS LOGIC MONITOR", style="Header.TLabel")
        title_lbl.pack(side=tk.LEFT)

        self.sys_status_var = tk.StringVar(value="STOPPED")
        self.sys_status_lbl = ttk.Label(header_frame, textvariable=self.sys_status_var, style="StatusWarn.TLabel", padding=5)
        self.sys_status_lbl.pack(side=tk.RIGHT)

        ttk.Separator(self.root, orient=tk.HORIZONTAL).pack(fill=tk.X, px=10)

        # Main Split
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left Column
        left_col = ttk.Frame(main_paned, padding=5)
        main_paned.add(left_col, weight=2)

        # Mode Selector & Controls Header
        mode_frame = ttk.LabelFrame(left_col, text="EXECUTION MODE & INPUT SOURCE", padding=8)
        mode_frame.pack(fill=tk.X, pady=4)

        ttk.Label(mode_frame, text="Input Mode:").pack(side=tk.LEFT, padx=3)
        self.input_mode_var = tk.StringVar(value="REPLAY")
        mode_combo = ttk.Combobox(mode_frame, textvariable=self.input_mode_var, values=["REPLAY", "LIVE_CAMERA", "MOCK"], width=14, state="readonly")
        mode_combo.pack(side=tk.LEFT, padx=5)
        mode_combo.bind("<<ComboboxSelected>>", self._on_mode_changed)

        ttk.Label(mode_frame, text="Cam Device:").pack(side=tk.LEFT, padx=(15, 2))
        self.cam_device_var = tk.StringVar(value="0")
        cam_combo = ttk.Combobox(mode_frame, textvariable=self.cam_device_var, values=["0", "1", "2"], width=4, state="readonly")
        cam_combo.pack(side=tk.LEFT, padx=3)
        cam_combo.bind("<<ComboboxSelected>>", self._on_cam_device_changed)

        self.btn_cam_start = ttk.Button(mode_frame, text="Start Camera", command=self._on_start_camera)
        self.btn_cam_start.pack(side=tk.LEFT, padx=5)

        self.btn_cam_stop = ttk.Button(mode_frame, text="Stop Camera", command=self._on_stop_camera)
        self.btn_cam_stop.pack(side=tk.LEFT, padx=3)

        # Status Cards Frame
        cards_frame = ttk.LabelFrame(left_col, text="EXPERIMENT STATUS", padding=10)
        cards_frame.pack(fill=tk.X, pady=4)

        # Grid layout for status cards
        ttk.Label(cards_frame, text="Current Step:", style="CardTitle.TLabel").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.step_var = tk.StringVar(value="1 / 15 (S01)")
        ttk.Label(cards_frame, textvariable=self.step_var, style="CardVal.TLabel").grid(row=0, column=1, sticky=tk.W, padx=10, pady=2)

        ttk.Label(cards_frame, text="Expected Action:", style="CardTitle.TLabel").grid(row=0, column=2, sticky=tk.W, pady=2)
        self.action_var = tk.StringVar(value="OPEN_WHITE_BOX")
        ttk.Label(cards_frame, textvariable=self.action_var, style="CardVal.TLabel").grid(row=0, column=3, sticky=tk.W, padx=10, pady=2)

        ttk.Label(cards_frame, text="Observed Action:", style="CardTitle.TLabel").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.obs_action_var = tk.StringVar(value="UNCERTAIN")
        ttk.Label(cards_frame, textvariable=self.obs_action_var, style="CardVal.TLabel").grid(row=1, column=1, sticky=tk.W, padx=10, pady=2)

        ttk.Label(cards_frame, text="Confidence:", style="CardTitle.TLabel").grid(row=1, column=2, sticky=tk.W, pady=2)
        self.conf_var = tk.StringVar(value="UNCERTAIN (0.00)")
        ttk.Label(cards_frame, textvariable=self.conf_var, style="CardVal.TLabel").grid(row=1, column=3, sticky=tk.W, padx=10, pady=2)

        ttk.Label(cards_frame, text="Next Step:", style="CardTitle.TLabel").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.next_step_var = tk.StringVar(value="S02 RETRIEVE_RED_BOX")
        ttk.Label(cards_frame, textvariable=self.next_step_var, style="CardVal.TLabel").grid(row=2, column=1, sticky=tk.W, padx=10, pady=2)

        ttk.Label(cards_frame, text="Status:", style="CardTitle.TLabel").grid(row=2, column=2, sticky=tk.W, pady=2)
        self.status_var = tk.StringVar(value="START")
        self.status_lbl = ttk.Label(cards_frame, textvariable=self.status_var, style="StatusGood.TLabel")
        self.status_lbl.grid(row=2, column=3, sticky=tk.W, padx=10, pady=2)

        # Real-time Telemetry Banner Frame
        metrics_frame = ttk.Frame(left_col, padding=5)
        metrics_frame.pack(fill=tk.X, pady=2)
        self.metrics_var = tk.StringVar(value="Cam FPS: 0.0 | P1 FPS: 0.0 | Proc FPS: 0.0 | Latency (Avg/P95/Max): 0.0 / 0.0 / 0.0 ms | Dropped: 0")
        ttk.Label(metrics_frame, textvariable=self.metrics_var, style="Metric.TLabel").pack(side=tk.LEFT)

        # Latest Alert Frame
        alert_frame = ttk.LabelFrame(left_col, text="LATEST ALERT / MESSAGE", padding=8)
        alert_frame.pack(fill=tk.X, pady=4)
        self.alert_var = tk.StringVar(value="Ready. Click 'START EXPERIMENT' or select dataset replay.")
        self.alert_lbl = ttk.Label(alert_frame, textvariable=self.alert_var, font=("Helvetica", 11, "italic"), wraplength=550)
        self.alert_lbl.pack(fill=tk.X)

        # Error Details Box
        self.error_frame = ttk.LabelFrame(left_col, text="PROCEDURAL ERROR DETAILS", padding=8)
        self.error_type_var = tk.StringVar(value="None")
        self.error_detail_var = tk.StringVar(value="")
        ttk.Label(self.error_frame, text="Error Type:", style="CardTitle.TLabel").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(self.error_frame, textvariable=self.error_type_var, style="Alert.TLabel").grid(row=0, column=1, sticky=tk.W, padx=10)
        ttk.Label(self.error_frame, textvariable=self.error_detail_var, font=("Helvetica", 10)).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=2)

        # Video Canvas Placeholder
        video_frame = ttk.LabelFrame(left_col, text="LIVE / REPLAY PERCEPTION FEED", padding=5)
        video_frame.pack(fill=tk.BOTH, expand=True, pady=4)
        self.video_canvas = tk.Canvas(video_frame, bg="#0f172a", height=200)
        self.video_canvas.pack(fill=tk.BOTH, expand=True)
        self.video_text_id = self.video_canvas.create_text(280, 100, text="No video source", fill="#94a3b8", font=("Helvetica", 14, "bold"))

        # Controls & Buttons Frame
        ctrl_frame = ttk.LabelFrame(left_col, text="EXPERIMENT CONTROLS", padding=8)
        ctrl_frame.pack(fill=tk.X, pady=4)

        file_sub_frame = ttk.Frame(ctrl_frame)
        file_sub_frame.pack(fill=tk.X, pady=2)
        ttk.Label(file_sub_frame, text="Replay Dataset:").pack(side=tk.LEFT)
        self.dataset_path_var = tk.StringVar(value=self.replay.dataset_path)
        ttk.Entry(file_sub_frame, textvariable=self.dataset_path_var, width=40).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_sub_frame, text="Browse", command=self._browse_file).pack(side=tk.LEFT)

        btn_sub_frame = ttk.Frame(ctrl_frame)
        btn_sub_frame.pack(fill=tk.X, pady=4)

        self.btn_exp_start = ttk.Button(btn_sub_frame, text="START EXPERIMENT", command=self._on_start_experiment)
        self.btn_exp_start.pack(side=tk.LEFT, padx=3)

        self.btn_pause = ttk.Button(btn_sub_frame, text="PAUSE", command=self._on_pause)
        self.btn_pause.pack(side=tk.LEFT, padx=3)

        self.btn_resume = ttk.Button(btn_sub_frame, text="RESUME", command=self._on_resume)
        self.btn_resume.pack(side=tk.LEFT, padx=3)

        self.btn_stop = ttk.Button(btn_sub_frame, text="STOP", command=self._on_stop)
        self.btn_stop.pack(side=tk.LEFT, padx=3)

        self.btn_reset = ttk.Button(btn_sub_frame, text="RESET", command=self._on_reset)
        self.btn_reset.pack(side=tk.LEFT, padx=3)

        ttk.Label(btn_sub_frame, text="Speed:").pack(side=tk.LEFT, padx=(10, 2))
        self.speed_var = tk.StringVar(value="1.0x")
        speed_combo = ttk.Combobox(btn_sub_frame, textvariable=self.speed_var, values=["0.25x", "0.5x", "1.0x", "2.0x", "4.0x"], width=5, state="readonly")
        speed_combo.pack(side=tk.LEFT)
        speed_combo.bind("<<ComboboxSelected>>", self._on_speed_changed)

        # Right Column
        right_col = ttk.Frame(main_paned, padding=5)
        main_paned.add(right_col, weight=2)

        # 15-Step Progress Tracker Frame
        steps_frame = ttk.LabelFrame(right_col, text="15-STEP EXPERIMENT PROGRESS", padding=8)
        steps_frame.pack(fill=tk.X, pady=4)

        self.step_items: Dict[str, ttk.Label] = {}
        steps_grid = ttk.Frame(steps_frame)
        steps_grid.pack(fill=tk.X)

        seq_list = [
            ("S01", "Open White Box"), ("S02", "Retrieve Red Box"), ("S03", "Retrieve Yellow Box"),
            ("S04", "Open Red Box"), ("S05", "Plant to Workplace"), ("S06", "Open Yellow Box"),
            ("S07", "Spray to Workplace"), ("S08", "Pick Spray Bottle"), ("S09", "Spray Plant"),
            ("S10", "Spray to Workplace"), ("S11", "Plant to Red Box"), ("S12", "Spray to Yellow Box"),
            ("S13", "Red Box to White Box"), ("S14", "Yellow Box to White Box"), ("S15", "Close White Box")
        ]

        for i, (sid, sname) in enumerate(seq_list):
            r = i % 8
            c = (i // 8) * 2
            lbl_id = ttk.Label(steps_grid, text=f"[{sid}]", font=("Helvetica", 9, "bold"), foreground="#64748b")
            lbl_id.grid(row=r, column=c, sticky=tk.W, padx=2, pady=1)
            lbl_txt = ttk.Label(steps_grid, text=sname, font=("Helvetica", 9), foreground="#334155")
            lbl_txt.grid(row=r, column=c + 1, sticky=tk.W, padx=5, pady=1)
            self.step_items[sid] = lbl_txt

        # System Health & Module Status Frame
        health_frame = ttk.LabelFrame(right_col, text="SYSTEM HEALTH & MANDATORY CAPABILITIES", padding=8)
        health_frame.pack(fill=tk.X, pady=4)

        h_grid = ttk.Frame(health_frame)
        h_grid.pack(fill=tk.X)

        # 4-Part Main Status Grid
        ttk.Label(h_grid, text="CAMERA:", font=("Helvetica", 9, "bold")).grid(row=0, column=0, sticky=tk.W)
        self.cam_status_var = tk.StringVar(value="STOPPED")
        ttk.Label(h_grid, textvariable=self.cam_status_var, font=("Helvetica", 9, "bold"), foreground="#ca8a04").grid(row=0, column=1, sticky=tk.W, padx=10)

        ttk.Label(h_grid, text="PERSON 1 AI:", font=("Helvetica", 9, "bold")).grid(row=0, column=2, sticky=tk.W)
        self.p1_status_var = tk.StringVar(value="NOT CONNECTED")
        ttk.Label(h_grid, textvariable=self.p1_status_var, font=("Helvetica", 9, "bold"), foreground="#dc2626").grid(row=0, column=3, sticky=tk.W, padx=10)

        ttk.Label(h_grid, text="PERSON 2:", font=("Helvetica", 9, "bold")).grid(row=1, column=0, sticky=tk.W)
        self.p2_status_var = tk.StringVar(value="STOPPED")
        ttk.Label(h_grid, textvariable=self.p2_status_var, font=("Helvetica", 9, "bold"), foreground="#ca8a04").grid(row=1, column=1, sticky=tk.W, padx=10)

        ttk.Label(h_grid, text="PERCEPTION:", font=("Helvetica", 9, "bold")).grid(row=1, column=2, sticky=tk.W)
        self.perception_status_var = tk.StringVar(value="UNAVAILABLE")
        ttk.Label(h_grid, textvariable=self.perception_status_var, font=("Helvetica", 9, "bold"), foreground="#dc2626").grid(row=1, column=3, sticky=tk.W, padx=10)

        # Recording & Streaming Info
        ttk.Label(h_grid, text="RECORDING:", font=("Helvetica", 9, "bold")).grid(row=2, column=0, sticky=tk.W)
        self.rec_status_var = tk.StringVar(value="OFF")
        ttk.Label(h_grid, textvariable=self.rec_status_var, font=("Helvetica", 9), foreground="#64748b").grid(row=2, column=1, sticky=tk.W, padx=10)

        ttk.Label(h_grid, text="IP STREAM:", font=("Helvetica", 9, "bold")).grid(row=2, column=2, sticky=tk.W)
        self.stream_status_var = tk.StringVar(value="OFF")
        ttk.Label(h_grid, textvariable=self.stream_status_var, font=("Helvetica", 9), foreground="#64748b").grid(row=2, column=3, sticky=tk.W, padx=10)

        # Active Recording File Display
        self.rec_file_var = tk.StringVar(value="None")
        ttk.Label(health_frame, text="Active Video Recording Path:", font=("Helvetica", 8, "bold"), foreground="#64748b").pack(anchor=tk.W, pady=(4, 0))
        ttk.Label(health_frame, textvariable=self.rec_file_var, font=("Helvetica", 8), foreground="#2563eb", wraplength=480).pack(anchor=tk.W)

        # Recent Event Log Table
        log_frame = ttk.LabelFrame(right_col, text="STRUCTURED EVENT LOG HISTORY", padding=8)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=4)

        cols = ("time", "frame", "event", "step", "status", "message")
        self.events_tree = ttk.Treeview(log_frame, columns=cols, show="headings", height=10)
        self.events_tree.heading("time", text="Time (s)")
        self.events_tree.heading("frame", text="Frame")
        self.events_tree.heading("event", text="Event")
        self.events_tree.heading("step", text="Step")
        self.events_tree.heading("status", text="Status")
        self.events_tree.heading("message", text="Message")

        self.events_tree.column("time", width=65)
        self.events_tree.column("frame", width=55)
        self.events_tree.column("event", width=120)
        self.events_tree.column("step", width=50)
        self.events_tree.column("status", width=90)
        self.events_tree.column("message", width=220)

        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.events_tree.yview)
        self.events_tree.configure(yscroll=scrollbar.set)
        self.events_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _on_mode_changed(self, event) -> None:
        mode = self.input_mode_var.get()
        if mode == "LIVE_CAMERA":
            self.live_camera.mock_camera = False
            self.adapter.mock_mode = False
            if not self.adapter.is_cv_model_connected:
                self.video_canvas.itemconfig(self.video_text_id, text="Person 1 perception model unavailable")
                self.p1_status_var.set("NOT CONNECTED")
                self.perception_status_var.set("UNAVAILABLE")
            else:
                self.video_canvas.itemconfig(self.video_text_id, text="Live Camera Active")
                self.p1_status_var.set("CONNECTED")
                self.perception_status_var.set("ACTIVE")
        elif mode == "MOCK":
            self.live_camera.mock_camera = True
            self.adapter.mock_mode = True
            self.video_canvas.itemconfig(self.video_text_id, text="Explicit MOCK Mode Active")
            self.p1_status_var.set("MOCK_ACTIVE")
            self.perception_status_var.set("ACTIVE")
        else:
            self.video_canvas.itemconfig(self.video_text_id, text="No video source")
            self.p1_status_var.set("REPLAY_MODE")
            self.perception_status_var.set("ACTIVE")

    def _on_cam_device_changed(self, event) -> None:
        idx = int(self.cam_device_var.get())
        self.live_camera.set_device_index(idx)

    def _on_start_camera(self) -> None:
        mode = self.input_mode_var.get()
        if mode == "REPLAY":
            self.input_mode_var.set("LIVE_CAMERA")
            self._on_mode_changed(None)

        success = self.live_camera.start_camera()
        if not success or self.live_camera.camera_status == "CAMERA_UNAVAILABLE":
            self.sys_status_var.set("CAMERA UNAVAILABLE")
            self.sys_status_lbl.configure(style="StatusErr.TLabel")
            self.cam_status_var.set("UNAVAILABLE")
            self.alert_var.set("Camera unavailable. Check webcam hardware connection.")
            messagebox.showerror("Camera Error", "Camera unavailable. Hardware device index could not be opened.")
            return

        self.cam_status_var.set("CONNECTED")
        self.sys_status_var.set("CAMERA ACTIVE")
        self.sys_status_lbl.configure(style="StatusGood.TLabel")

        if not self.adapter.is_cv_model_connected and mode == "LIVE_CAMERA":
            self.p1_status_var.set("NOT CONNECTED")
            self.perception_status_var.set("UNAVAILABLE")
            self.alert_var.set("Camera connected. WARNING: Person 1 perception model is missing or not connected.")
        else:
            self.alert_var.set("Camera started. Click 'START EXPERIMENT' to initialize execution tracking.")

    def _on_stop_camera(self) -> None:
        self.live_camera.stop_camera()
        self.sys_status_var.set("CAMERA STOPPED")
        self.sys_status_lbl.configure(style="StatusWarn.TLabel")
        self.cam_status_var.set("STOPPED")
        self.p2_status_var.set("STOPPED")
        self.rec_status_var.set("OFF")
        self.stream_status_var.set("OFF")
        self.alert_var.set("Camera and recording stopped.")

    def _browse_file(self) -> None:
        filename = filedialog.askopenfilename(
            title="Select Person 1 Visual Output File",
            filetypes=[("JSON Lines Files", "*.jsonl"), ("All Files", "*.*")]
        )
        if filename:
            self.dataset_path_var.set(filename)
            self.replay.dataset_path = filename

    def _on_start_experiment(self) -> None:
        mode = self.input_mode_var.get()
        if mode == "REPLAY":
            self.replay.dataset_path = self.dataset_path_var.get()
            self.sys_status_var.set("REPLAY RUNNING")
            self.sys_status_lbl.configure(style="StatusGood.TLabel")
            self.p2_status_var.set("ACTIVE")
            self.alert_var.set("Replay experiment started.")
            self.replay.start()
        elif mode == "LIVE_CAMERA":
            if not self.adapter.is_cv_model_connected:
                self.sys_status_var.set("PERCEPTION UNAVAILABLE")
                self.sys_status_lbl.configure(style="StatusErr.TLabel")
                self.alert_var.set("Person 1 perception model unavailable — cannot start live experiment.")
                messagebox.showwarning(
                    "Perception Unavailable",
                    "Person 1 perception model is not connected.\n\n"
                    "Please provide Person 1 model files in config/person1_live.json or switch to 'MOCK' mode to test downstream logic."
                )
                return

            self.live_camera.start_experiment()
            self.sys_status_var.set("EXPERIMENT ACTIVE")
            self.sys_status_lbl.configure(style="StatusGood.TLabel")
            self.p2_status_var.set("ACTIVE")
            self.rec_status_var.set("ON" if self.recorder.is_recording else "OFF")
            self.rec_file_var.set(self.recorder.recording_path or "None")
            self.alert_var.set("Live experiment started. Recording & monitoring active.")
        else:  # MOCK Mode
            self.live_camera.start_experiment()
            self.sys_status_var.set("MOCK EXPERIMENT ACTIVE")
            self.sys_status_lbl.configure(style="StatusGood.TLabel")
            self.p2_status_var.set("ACTIVE")
            self.rec_status_var.set("ON" if self.recorder.is_recording else "OFF")
            self.rec_file_var.set(self.recorder.recording_path or "None")
            self.alert_var.set("Explicit MOCK experiment started. Monitoring S01...")

    def _on_pause(self) -> None:
        if self.input_mode_var.get() == "REPLAY":
            self.replay.pause()
        else:
            self.live_camera.pause_experiment()
        self.sys_status_var.set("PAUSED")
        self.sys_status_lbl.configure(style="StatusWarn.TLabel")
        self.p2_status_var.set("PAUSED")
        self.alert_var.set("Experiment paused.")

    def _on_resume(self) -> None:
        if self.input_mode_var.get() == "REPLAY":
            self.replay.resume()
        else:
            self.live_camera.resume_experiment()
        self.sys_status_var.set("RUNNING")
        self.sys_status_lbl.configure(style="StatusGood.TLabel")
        self.p2_status_var.set("ACTIVE")
        self.alert_var.set("Experiment resumed.")

    def _on_stop(self) -> None:
        if self.input_mode_var.get() == "REPLAY":
            self.replay.stop()
        else:
            self.live_camera.stop_camera()
        self.sys_status_var.set("STOPPED")
        self.sys_status_lbl.configure(style="StatusWarn.TLabel")
        self.p2_status_var.set("STOPPED")
        self.rec_status_var.set("OFF")
        self.alert_var.set("Experiment stopped.")

    def _on_reset(self) -> None:
        if self.input_mode_var.get() == "REPLAY":
            self.replay.reset()
        else:
            self.live_camera.reset_experiment()

        self.sys_status_var.set("STOPPED")
        self.sys_status_lbl.configure(style="StatusWarn.TLabel")
        self.step_var.set("1 / 15 (S01)")
        self.action_var.set("OPEN_WHITE_BOX")
        self.obs_action_var.set("UNCERTAIN")
        self.conf_var.set("UNCERTAIN (0.00)")
        self.next_step_var.set("S02 RETRIEVE_RED_BOX")
        self.status_var.set("START")
        self.alert_var.set("System reset. Ready to start.")
        self.metrics_var.set("Cam FPS: 0.0 | P1 FPS: 0.0 | Proc FPS: 0.0 | Latency (Avg/P95/Max): 0.0 / 0.0 / 0.0 ms | Dropped: 0")
        self.rec_status_var.set("OFF")
        self.rec_file_var.set("None")
        self.error_frame.pack_forget()

        for sid, lbl in self.step_items.items():
            lbl.configure(font=("Helvetica", 9), foreground="#334155")

        for item in self.events_tree.get_children():
            self.events_tree.delete(item)

    def _on_speed_changed(self, event) -> None:
        val = self.speed_var.get().replace("x", "")
        try:
            sp = float(val)
            self.replay.set_speed(sp)
        except ValueError:
            pass

    def _queue_update_callback(self, update: StateUpdate, frame_obj: NormalizedFrame, obs_action: ObservedAction) -> None:
        self.update_queue.put(("UPDATE", update, frame_obj, obs_action, None))

    def _queue_live_frame_callback(self, cv_image: Any, update: StateUpdate, frame_obj: NormalizedFrame, obs_action: ObservedAction, metrics: Dict[str, Any]) -> None:
        self.update_queue.put(("LIVE_FRAME", update, frame_obj, obs_action, metrics))

    def _queue_complete_callback(self) -> None:
        self.update_queue.put(("COMPLETE", None, None, None, None))

    def _start_queue_listener(self) -> None:
        self._process_queue()

    def _process_queue(self) -> None:
        try:
            while True:
                msg_type, update, frame_obj, obs_action, metrics = self.update_queue.get_nowait()
                if msg_type in ("UPDATE", "LIVE_FRAME"):
                    self._apply_update(update, frame_obj, obs_action, metrics)
                elif msg_type == "COMPLETE":
                    self.sys_status_var.set("COMPLETE")
                    self.sys_status_lbl.configure(style="StatusGood.TLabel")
                    self.alert_var.set("Experiment complete.")

                self.update_queue.task_done()
        except queue.Empty:
            pass

        self.root.after(40, self._process_queue)

    def _apply_update(self, update: StateUpdate, frame_obj: NormalizedFrame, obs_action: ObservedAction, metrics: Optional[Dict[str, Any]] = None) -> None:
        self.step_var.set(f"{update.current_step} / 15 ({update.current_state_id})")
        self.action_var.set(update.expected_action)
        self.obs_action_var.set(obs_action.action)
        self.conf_var.set(f"{obs_action.confidence_level} ({obs_action.confidence:.2f})")
        self.next_step_var.set(f"{update.next_step or 'None'}")
        self.status_var.set(update.status)
        self.alert_var.set(update.message)

        if metrics:
            self.metrics_var.set(
                f"Cam FPS: {metrics.get('camera_fps', 0.0)} | P1 FPS: {metrics.get('person1_fps', 0.0)} | "
                f"Proc FPS: {metrics.get('processing_fps', 0.0)} | "
                f"Latency (Avg/P95/Max): {metrics.get('avg_latency_ms', 0.0)} / {metrics.get('p95_latency_ms', 0.0)} / {metrics.get('max_latency_ms', 0.0)} ms | "
                f"Dropped: {metrics.get('dropped_frames', 0)}"
            )

            self.cam_status_var.set(metrics.get("camera_status", "CONNECTED"))
            self.p1_status_var.set(metrics.get("p1_status", "NOT CONNECTED"))
            self.rec_status_var.set("ON" if metrics.get("recording_active") else "OFF")
            if metrics.get("recording_path"):
                self.rec_file_var.set(metrics["recording_path"])
            stream_stat = metrics.get("stream_status", "OFF")
            if stream_stat == "CONNECTED":
                self.stream_status_var.set(metrics.get("stream_url", "CONNECTED"))
            else:
                self.stream_status_var.set(stream_stat)

        if update.status == "ERROR":
            self.status_lbl.configure(style="StatusErr.TLabel")
            self.error_frame.pack(fill=tk.X, pady=4)
            self.error_type_var.set(update.error_type or "PROCEDURAL_ERROR")
            self.error_detail_var.set(f"Expected: {update.expected_action} | Observed: {update.observed_action}\n{update.message}")
        else:
            self.status_lbl.configure(style="StatusGood.TLabel" if update.status in ("ACTIVE", "STEP_COMPLETED", "COMPLETE") else "StatusWarn.TLabel")
            self.error_frame.pack_forget()

        # Update step tracker highlight
        for sid, lbl in self.step_items.items():
            if sid == update.current_state_id:
                lbl.configure(font=("Helvetica", 9, "bold"), foreground="#2563eb")
            else:
                lbl.configure(font=("Helvetica", 9), foreground="#334155")

        # Update events log tree if meaningful event occurred
        if update.status in ("STEP_COMPLETED", "ERROR", "RECOVERY", "COMPLETE") or update.transitioned:
            evt_name = update.status if not update.error_type else update.error_type
            self.events_tree.insert(
                "", 0,
                values=(
                    f"{frame_obj.timestamp:.2f}",
                    frame_obj.frame,
                    evt_name,
                    update.current_state_id,
                    update.status,
                    update.message[:50]
                )
            )
