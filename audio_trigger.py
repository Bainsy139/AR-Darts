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
import sys

# ----------------------------
# Configuration (tuned values)
# ----------------------------
AUDIO_DEVICE_INDEX = 0      # PyAudio index for USB PnP Sound Device
SAMPLE_RATE = 44100
CHUNK = 1024                # frames per buffer
RMS_THRESHOLD = 0.09         # trigger threshold
COOLDOWN_SEC = 2.0          # minimum time between triggers
SETTLE_SEC = 0.30           # wait after impact before detection (dart vibration settle)
RETRY_ON_NO_IMPACT = True   # one retry if vision says "no_impact"
RETRY_DELAY_SEC = 0.20      # extra wait before retry
DEBUG_RMS = False            # set True to print live RMS (set False once threshold is dialled in)
SERVER_BASE = "http://127.0.0.1:5050"

CAPTURE_BEFORE_ENDPOINT = f"{SERVER_BASE}/capture-before"
DETECT_ENDPOINT = f"{SERVER_BASE}/detect"

# ----------------------------
# Internal state
# ----------------------------
last_trigger_time = 0.0
_busy = threading.Event()  # replaces busy_handling_spike + spike_lock

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
    """POST helper with basic error handling. Returns (ok, json_data_or_none)."""
    try:
        r = requests.post(endpoint, timeout=15)
        if r.status_code != 200:
            print(f"[WARN] {endpoint} -> HTTP {r.status_code}")
            return False, None
        try:
            return True, r.json()
        except requests.exceptions.JSONDecodeError:
            print(f"[WARN] {endpoint} -> Received non-JSON response with HTTP 200")
            return True, None
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] POST {endpoint}: {e}")
        return False, None

def _handle_spike_blocking_actions():
    print("[SPIKE] Dart impact detected (processing in background)...")
    try:
        time.sleep(SETTLE_SEC)
        ok, data = post(DETECT_ENDPOINT)

        if RETRY_ON_NO_IMPACT and ok and isinstance(data, dict):
            reason = data.get("reason") or (data.get("hit") or {}).get("reason")
            if reason == "no_impact":
                print("[SPIKE] No impact detected by vision, retrying...")
                time.sleep(RETRY_DELAY_SEC)
                post(DETECT_ENDPOINT)

        print("[SPIKE] Refreshing baseline for next dart...")
        post(CAPTURE_BEFORE_ENDPOINT)

    except Exception as e:
        print(f"[ERROR] Spike handling error: {e}")
    finally:
        _busy.clear()  # always release, even on error
        print("[SPIKE] Spike handling complete.")


def handle_spike_non_blocking():
    global last_trigger_time
    now = time.time()
    if (now - last_trigger_time) < COOLDOWN_SEC:
        return
    if _busy.is_set():
        return
    _busy.set()  # set immediately — no second check needed
    last_trigger_time = now
    spike_thread = threading.Thread(target=_handle_spike_blocking_actions)
    spike_thread.daemon = True
    spike_thread.start()

# ----------------------------
# Main loop
# ----------------------------
def main():
    print("[AUDIO] Starting audio trigger listener")
    pa = None
    stream = None
    try:
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

        # Capture initial baseline
        print("[INIT] Capturing baseline frame...")
        ok, _ = post(CAPTURE_BEFORE_ENDPOINT)
        if not ok:
            print("[ERROR] Initial baseline capture failed. Check Flask server.")
            sys.exit(1)

        print("[INIT] Baseline captured. Ready — throw a dart!")

        while True:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                rms = rms_from_bytes(data)
                if DEBUG_RMS:
                    print(f"RMS={rms:.4f}")

                if rms >= RMS_THRESHOLD:
                    handle_spike_non_blocking()

            except IOError as e:
                print(f"[ERROR] Audio stream read error: {e}")
                time.sleep(1)
            except Exception as e:
                print(f"[ERROR] Unexpected error in audio loop: {e}")
                break

    except KeyboardInterrupt:
        print("\n[AUDIO] Stopping listener due to KeyboardInterrupt.")
    except Exception as e:
        print(f"[CRITICAL ERROR] PyAudio failed to initialize or open stream: {e}")
        print("Please check your microphone connection and configuration.")
        sys.exit(1)
    finally:
        print("[AUDIO] Cleaning up audio resources...")
        if stream and stream.is_active():
            stream.stop_stream()
        if stream:
            stream.close()
        if pa:
            pa.terminate()
        print("[AUDIO] Audio resources cleaned up.")

if __name__ == "__main__":
    main()
