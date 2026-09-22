"""
Verification script for IP Video Streaming (Part A).
Tests localhost access (TEST 1) and LAN IP interface access (TEST 2).
"""
import os
import sys
import time
import urllib.request
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath('.'))
from src.streaming import IPStreamer, get_lan_ip

def test_streaming():
    print("========================================")
    print("PART A — IP VIDEO STREAMING VERIFICATION")
    print("========================================")
    
    lan_ip = get_lan_ip()
    print(f"Discovered Host LAN IP: {lan_ip}")
    
    streamer = IPStreamer(config_path="config/streaming.json")
    streamer.config["stream_enabled"] = True
    streamer.config["stream_host"] = "0.0.0.0"
    streamer.config["stream_port"] = 8554
    
    started = streamer.start_stream()
    if not started:
        print(f"FAILED to start streamer server: {streamer.error_message}")
        return False

    stream_url = streamer.get_stream_url()
    print(f"Streaming server active at: {stream_url}")
    
    # Push test frames
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(dummy_img, "SIH TEST STREAM", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    streamer.push_frame(dummy_img)
    time.sleep(0.2)
    
    # TEST 1: Same machine access (localhost)
    test1_passed = False
    try:
        req = urllib.request.urlopen("http://127.0.0.1:8554/stream", timeout=2.0)
        header = req.headers.get("Content-Type")
        if req.status == 200 and "multipart/x-mixed-replace" in str(header):
            print("TEST 1 PASSED: Localhost (127.0.0.1:8554/stream) HTTP 200 MJPEG stream accessible.")
            test1_passed = True
        else:
            print(f"TEST 1 FAILED: Invalid response headers: {header}")
    except Exception as e:
        print(f"TEST 1 FAILED: {e}")

    # TEST 2: LAN interface access
    test2_passed = False
    try:
        lan_url = f"http://{lan_ip}:8554/stream"
        req = urllib.request.urlopen(lan_url, timeout=2.0)
        header = req.headers.get("Content-Type")
        if req.status == 200 and "multipart/x-mixed-replace" in str(header):
            print(f"TEST 2 PASSED: LAN Interface ({lan_url}) HTTP 200 MJPEG stream accessible across network.")
            test2_passed = True
        else:
            print(f"TEST 2 FAILED: Invalid response headers: {header}")
    except Exception as e:
        print(f"TEST 2 FAILED: {e}")

    streamer.stop_stream()
    print("========================================")
    print(f"IP STREAMING RESULT: TEST 1={test1_passed}, TEST 2={test2_passed}")
    return test1_passed and test2_passed

if __name__ == "__main__":
    test_streaming()
