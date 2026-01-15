#!/usr/bin/env python3
"""
audio_trigger.py
Standalone audio spike listener for AR Darts.

Responsibilities:
- Listen to USB microphone
- Detect RMS spikes above threshold
- Enforce cooldown + busy latch
- Call Flask endpoints:
    POST /capture-before
    POST /detect
"""

import time
import math
import struct
import threading
import requests
import pyaudio

# ----------------------------
# Configuration (tuned values)
# ----------------------------
AUDIO_DEVICE_INDEX = None   # None = default input (USB mic already set as capture)
SAMPLE_RATE = 44100
CHUNK = 1024                # frames per buffer
RMS_THRESHOLD = 0.008       # trigger threshold
COOLDOWN_SEC = 0.4          # minimum time between triggers
SERVER_BASE = "http://127.0.0.1:5050"

CAPTURE_BEFORE_ENDPOINT = f"{SERVER_BASE}/capture-before"
DETECT_ENDPOINT = f"{SERVER_BASE}/detect"

# ----------------------------
# Internal state
# ----------------------------
last_trigger_time = 0.0
busy = False
lock = threading.Lock()

# ----------------------------
# Helpers
# ----------------------------
def rms_from_bytes(data: bytes) -> float:
    """Compute RMS from 16-bit PCM audio chunk."""
    count = len(data) // 2
    if count == 0:
        return 0.0
    samples = struct.unpack(f"{count}h", data)
    square_sum = 0.0
    for s in samples:
        norm = s / 32768.0
        square_sum += norm * norm
    return math.sqrt(square_sum / count)

def post(endpoint: str):
    """POST helper with basic error handling."""
    try:
        r = requests.post(endpoint, timeout=2)
        if r.status_code != 200:
            print(f"[WARN] {endpoint} -> {r.status_code}")
    except Exception as e:
        print(f"[ERROR] POST {endpoint}: {e}")

def handle_spike():
    global busy
    with lock:
        if busy:
            return
        busy = True

    print("[SPIKE] Dart impact detected")

    # Run detect, then immediately capture-before again
    post(DETECT_ENDPOINT)
    post(CAPTURE_BEFORE_ENDPOINT)

    with lock:
        busy = False

# ----------------------------
# Main loop
# ----------------------------
def main():
    global last_trigger_time

    print("[AUDIO] Starting audio trigger listener")

    pa = pyaudio.PyAudio()

    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        input_device_index=AUDIO_DEVICE_INDEX,
        frames_per_buffer=CHUNK,
    )

    print("[AUDIO] Mic stream open, listening for spikes")

    # Initial capture-before at game start
    print("[INIT] Capturing baseline frame")
    post(CAPTURE_BEFORE_ENDPOINT)

    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            rms = rms_from_bytes(data)
            print(f"RMS={rms:.4f}")

            now = time.time()
            if rms >= RMS_THRESHOLD and (now - last_trigger_time) >= COOLDOWN_SEC:
                last_trigger_time = now
                handle_spike()

    except KeyboardInterrupt:
        print("\n[AUDIO] Stopping listener")

    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

if __name__ == "__main__":
    main()