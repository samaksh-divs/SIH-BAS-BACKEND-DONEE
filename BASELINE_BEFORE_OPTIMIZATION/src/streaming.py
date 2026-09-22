"""
IP Video Streaming Module (Phase 12)
Streams live camera frames asynchronously over HTTP MJPEG.
Configurable via config/streaming.json.
Fault-isolated: Network or stream errors NEVER stop perception, state machine, logging, or recording.
"""
import os
import json
import time
import queue
import threading
import socket
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Dict, Any, Optional

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def get_lan_ip() -> str:
    """Helper to discover LAN IP address of local network interface."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.1)
        s.connect(('10.255.255.255', 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        try:
            ip = socket.gethostbyname(socket.gethostname())
            if ip and not ip.startswith("127."):
                return ip
            return "127.0.0.1"
        except Exception:
            return "127.0.0.1"


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Threaded HTTP Server for handling concurrent stream clients."""
    daemon_threads = True


class MJPEGStreamHandler(BaseHTTPRequestHandler):
    """HTTP Handler serving MJPEG multipart video stream."""
    streamer_ref = None

    def do_GET(self):
        if self.path in ('/', '/stream', '/video'):
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.end_headers()

            while self.streamer_ref and self.streamer_ref.is_streaming:
                jpeg_bytes = self.streamer_ref.get_latest_jpeg()
                if jpeg_bytes:
                    try:
                        self.wfile.write(b'--frame\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', str(len(jpeg_bytes)))
                        self.end_headers()
                        self.wfile.write(jpeg_bytes)
                        self.wfile.write(b'\r\n')
                        time.sleep(0.03)
                    except Exception:
                        break
                else:
                    time.sleep(0.05)
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        # Suppress HTTP access logging to keep console output clean
        pass


class IPStreamer:
    """
    Asynchronous IP video streamer using HTTP MJPEG.
    """

    def __init__(self, config_path: str = "config/streaming.json"):
        self.config_path = config_path
        self.config: Dict[str, Any] = {
            "stream_enabled": False,
            "stream_host": "0.0.0.0",
            "stream_port": 8554,
            "stream_protocol": "mjpeg"
        }
        self.is_streaming: bool = False
        self.status: str = "DISCONNECTED"
        self.error_message: Optional[str] = None
        self._latest_jpeg: Optional[bytes] = None

        self._server: Optional[HTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None

        self._load_config()

    def _load_config(self) -> None:
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    user_cfg = json.load(f)
                    self.config.update(user_cfg)
            except Exception:
                pass

    def get_stream_url(self) -> str:
        """Returns full HTTP URL for accessing the video stream."""
        host = self.config.get("stream_host", "0.0.0.0")
        port = int(self.config.get("stream_port", 8554))
        if host in ("0.0.0.0", "::", ""):
            ip = get_lan_ip()
            return f"http://{ip}:{port}/stream"
        return f"http://{host}:{port}/stream"

    def start_stream(self) -> bool:
        """Starts asynchronous HTTP MJPEG streaming server."""
        if self.is_streaming:
            return True

        host = self.config.get("stream_host", "0.0.0.0")
        port = int(self.config.get("stream_port", 8554))

        MJPEGStreamHandler.streamer_ref = self

        try:
            self._server = ThreadedHTTPServer((host, port), MJPEGStreamHandler)
            self.is_streaming = True
            self.status = "CONNECTED"
            self.error_message = None

            self._server_thread = threading.Thread(target=self._run_server, daemon=True)
            self._server_thread.start()
            print(f"[IPStreamer] Server started on {host}:{port} -> Access stream at {self.get_stream_url()}")
            return True
        except Exception as e:
            self.is_streaming = False
            self.status = "ERROR"
            self.error_message = str(e)
            return False

    def _run_server(self) -> None:
        """Background thread handling HTTP requests."""
        if self._server:
            try:
                self._server.serve_forever()
            except Exception as e:
                self.status = "ERROR"
                self.error_message = str(e)
                self.is_streaming = False

    def push_frame(self, cv_image: Any) -> bool:
        """
        Converts BGR image matrix to JPEG bytes asynchronously for streaming.
        Fault-isolated: fails gracefully without throwing.
        """
        if not self.is_streaming or cv_image is None:
            return False

        try:
            if HAS_CV2:
                ret, buffer = cv2.imencode('.jpg', cv_image, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                if ret:
                    self._latest_jpeg = buffer.tobytes()
                    return True
        except Exception as e:
            self.status = "ERROR"
            self.error_message = str(e)
        return False

    def get_latest_jpeg(self) -> Optional[bytes]:
        return self._latest_jpeg

    def stop_stream(self) -> None:
        """Stops the streaming server."""
        self.is_streaming = False
        self.status = "DISCONNECTED"

        if self._server:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:
                pass
            self._server = None
        self._server_thread = None

    def close(self) -> None:
        self.stop_stream()
