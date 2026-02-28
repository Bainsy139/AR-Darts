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
RMS_THRESHOLD = 0.09        # trigger threshold
COOLDOWN_SEC = 2.0          # minimum time between triggers
SETTLE_SEC = 0.30           # wait after impact before detection (dart vibration settle)
RETRY_ON_NO_IMPACT = True   # one retry if vision says "no_impact"
RETRY_DELAY_SEC = 0.20      # extra wait before retry
DEBUG_RMS = False           # set True to print live RMS
MAX_BUSY_SEC = 12.0         # watchdog: force-release _busy after this many seconds
SERVER_BASE = "http://127.0.0.1:5050"

CAPTURE_BEFORE_ENDPOINT = f"{SERVER_BASE}/capture-before"
DETECT_ENDPOINT         = f"{SERVER_BASE}/detect"

# ----------------------------
# Internal state
# ----------------------------
last_trigger_time = 0.0
_busy = threading.Event()   # set = currently handling a spike

# ----------------------------
# Helpers
# ----------------------------
def rms_from_bytes(data: bytes) -> float:
    """Compute RMS from 16-bit PCM audio chunk."""
    count = len(data) // 2
    if count == 0:
        return 0.0
    samples = struct.unpack(f"{count}h", data)
    square_sum = sum((s / 32768.0) ** 2 for s in samples)
    return math.sqrt(square_sum / count)

def post(endpoint: str, timeout: int = 10):
    """POST helper. Returns (ok, json_data_or_none)."""
    try:
        r = requests.post(endpoint, timeout=timeout)
        if r.status_code != 200:
            print(f"[WARN] {endpoint} -> HTTP {r.status_code}")
            return False, None
        try:
            return True, r.json()
        except requests.exceptions.JSONDecodeError:
            print(f"[WARN] {endpoint} -> non-JSON response")
            return True, None
    except requests.exceptions.Timeout:
        print(f"[WARN] POST {endpoint} timed out after {timeout}s")
        return False, None
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] POST {endpoint}: {e}")
        return False, None

def _handle_spike():
    """Runs in a background thread. Always releases _busy in finally."""
    print("[SPIKE] Dart impact detected — processing...")
    busy_since = time.time()

    try:
        time.sleep(SETTLE_SEC)

        # Watchdog check before detect
        if time.time() - busy_since > MAX_BUSY_SEC:
            print("[WARN] Watchdog: spike handler taking too long, aborting.")
            return

        ok, data = post(DETECT_ENDPOINT)

        if RETRY_ON_NO_IMPACT and ok and isinstance(data, dict):
            reason = data.get("reason") or (data.get("hit") or {}).get("reason")
            if reason == "no_impact":
                print("[SPIKE] No impact from vision — retrying...")
                time.sleep(RETRY_DELAY_SEC)
                post(DETECT_ENDPOINT)

        print("[SPIKE] Refreshing baseline...")
        post(CAPTURE_BEFORE_ENDPOINT)

    except Exception as e:
        print(f"[ERROR] Spike handler exception: {e}")

    finally:
        _busy.clear()  # ALWAYS released — no matter what
        elapsed = time.time() - busy_since
        print(f"[SPIKE] Done. ({elapsed:.1f}s)")


def handle_spike_non_blocking():
    """Called from audio loop. Fires spike handler if not busy and cooldown elapsed."""
    global last_trigger_time
    now = time.time()

    # Cooldown check
    if (now - last_trigger_time) < COOLDOWN_SEC:
        return

    # Watchdog: if _busy has been set for too long, force-clear it
    if _busy.is_set():
        busy_age = now - last_trigger_time
        if busy_age > MAX_BUSY_SEC:
            print(f"[WARN] Watchdog: _busy stuck for {busy_age:.1f}s — force-releasing!")
            _busy.clear()
        else:
            return  # legitimately busy, skip

    # Claim the latch atomically
    _busy.set()
    last_trigger_time = now

    t = threading.Thread(target=_handle_spike, daemon=True)
    t.start()


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

        print("[AUDIO] Mic stream open — listening for spikes")

        print("[INIT] Capturing initial baseline...")
        ok, _ = post(CAPTURE_BEFORE_ENDPOINT)
        if not ok:
            print("[ERROR] Initial baseline failed. Is Flask running?")
            sys.exit(1)
        print("[INIT] Baseline ready. Throw a dart!")

        while True:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                rms = rms_from_bytes(data)
                if DEBUG_RMS:
                    print(f"RMS={rms:.4f}")
                if rms >= RMS_THRESHOLD:
                    handle_spike_non_blocking()

            except IOError as e:
                print(f"[ERROR] Audio read error: {e}")
                time.sleep(0.5)
            except Exception as e:
                print(f"[ERROR] Unexpected loop error: {e}")
                # continue rather than break — keep the mic alive
                time.sleep(0.5)
                continue

    except KeyboardInterrupt:
        print("\n[AUDIO] Stopped by user.")
    except Exception as e:
        print(f"[CRITICAL] PyAudio init failed: {e}")
        sys.exit(1)
    finally:
        print("[AUDIO] Cleaning up...")
        try:
            if stream and stream.is_active():
                stream.stop_stream()
            if stream:
                stream.close()
            if pa:
                pa.terminate()
        except Exception:
            pass
        print("[AUDIO] Done.")

if __name__ == "__main__":
    main()