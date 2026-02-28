#!/usr/bin/env python3
"""
audio_trigger.py
Standalone audio spike listener for AR Darts.

Responsibilities:
- Listen to USB microphone
- Detect RMS spikes above threshold
- Enforce cooldown + busy latch
- Poll /audio-armed to know when to listen
- Auto-disarm after 3 darts
- Re-armed by JS when player presses Ready (via /arm-audio endpoint on Flask)
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
AUDIO_DEVICE_INDEX = 0
SAMPLE_RATE        = 44100
CHUNK              = 1024
RMS_THRESHOLD      = 0.09
COOLDOWN_SEC       = 2.0
SETTLE_SEC         = 0.30
RETRY_ON_NO_IMPACT = True
RETRY_DELAY_SEC    = 0.20
DEBUG_RMS          = False
MAX_BUSY_SEC       = 12.0
DARTS_PER_TURN     = 3        # disarm after this many successful detections

SERVER_BASE = "http://127.0.0.1:5050"
CAPTURE_BEFORE_ENDPOINT = f"{SERVER_BASE}/capture-before"
DETECT_ENDPOINT         = f"{SERVER_BASE}/detect"
ARM_STATUS_ENDPOINT     = f"{SERVER_BASE}/audio-armed"   # GET — returns {"armed": true/false}

# ----------------------------
# Internal state
# ----------------------------
last_trigger_time = 0.0
_busy             = threading.Event()
_armed            = threading.Event()   # set = listening for darts
_dart_count       = 0                   # darts detected this turn
_dart_count_lock  = threading.Lock()

# ----------------------------
# Helpers
# ----------------------------
def rms_from_bytes(data: bytes) -> float:
    count = len(data) // 2
    if count == 0:
        return 0.0
    samples = struct.unpack(f"{count}h", data)
    square_sum = sum((s / 32768.0) ** 2 for s in samples)
    return math.sqrt(square_sum / count)

def post(endpoint: str, timeout: int = 10):
    try:
        r = requests.post(endpoint, timeout=timeout)
        if r.status_code != 200:
            print(f"[WARN] {endpoint} -> HTTP {r.status_code}")
            return False, None
        try:
            return True, r.json()
        except requests.exceptions.JSONDecodeError:
            return True, None
    except requests.exceptions.Timeout:
        print(f"[WARN] POST {endpoint} timed out after {timeout}s")
        return False, None
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] POST {endpoint}: {e}")
        return False, None

def get(endpoint: str, timeout: int = 3):
    try:
        r = requests.get(endpoint, timeout=timeout)
        if r.status_code == 200:
            return True, r.json()
        return False, None
    except Exception:
        return False, None

# ----------------------------
# Arm state poller
# Polls Flask /audio-armed every second.
# When Flask says armed=true, sets _armed and resets dart count.
# ----------------------------
def _poll_arm_state():
    global _dart_count
    last_state = None
    while True:
        ok, data = get(ARM_STATUS_ENDPOINT)
        if ok and isinstance(data, dict):
            armed = data.get("armed", False)
            if armed and not _armed.is_set():
                print("[ARM] Armed — capturing fresh baseline, ready for player...")
                post(CAPTURE_BEFORE_ENDPOINT)
                with _dart_count_lock:
                    _dart_count = 0
                _busy.clear()
                _armed.set()
                print("[ARM] Listening for darts!")
            elif not armed and _armed.is_set():
                _armed.clear()
                print("[ARM] Disarmed — waiting for Ready button.")
        time.sleep(0.5)

# ----------------------------
# Spike handler
# ----------------------------
def _handle_spike():
    global _dart_count
    print("[SPIKE] Dart impact detected — processing...")
    busy_since = time.time()

    try:
        time.sleep(SETTLE_SEC)

        if time.time() - busy_since > MAX_BUSY_SEC:
            print("[WARN] Watchdog: aborting slow spike handler.")
            return

        ok, data = post(DETECT_ENDPOINT)

        if RETRY_ON_NO_IMPACT and ok and isinstance(data, dict):
            reason = data.get("reason") or (data.get("hit") or {}).get("reason")
            if reason == "no_impact":
                print("[SPIKE] No impact — retrying...")
                time.sleep(RETRY_DELAY_SEC)
                ok, data = post(DETECT_ENDPOINT)

        # Only count as a dart if we got a valid hit
        hit = data.get("hit") if (ok and isinstance(data, dict)) else None
        if hit:
            with _dart_count_lock:
                _dart_count += 1
                count = _dart_count
            print(f"[SPIKE] Dart {count}/{DARTS_PER_TURN} detected.")

            if count < DARTS_PER_TURN:
                # More darts to come — refresh baseline for next dart
                print("[SPIKE] Refreshing baseline for next dart...")
                post(CAPTURE_BEFORE_ENDPOINT)
            else:
                # Turn over — disarm and wait for Ready
                print(f"[SPIKE] {DARTS_PER_TURN} darts thrown — disarming. Waiting for Ready.")
                _armed.clear()
                # DO NOT capture-before here — Ready button will do it
        else:
            # No valid hit — still refresh baseline and keep listening
            print("[SPIKE] No valid hit scored — refreshing baseline, still armed.")
            post(CAPTURE_BEFORE_ENDPOINT)

    except Exception as e:
        print(f"[ERROR] Spike handler: {e}")

    finally:
        _busy.clear()
        elapsed = time.time() - busy_since
        print(f"[SPIKE] Done. ({elapsed:.1f}s)")


def handle_spike_non_blocking():
    global last_trigger_time
    now = time.time()

    if not _armed.is_set():
        return

    if (now - last_trigger_time) < COOLDOWN_SEC:
        return

    if _busy.is_set():
        busy_age = now - last_trigger_time
        if busy_age > MAX_BUSY_SEC:
            print(f"[WARN] Watchdog: force-releasing stuck _busy ({busy_age:.1f}s)")
            _busy.clear()
        else:
            return

    _busy.set()
    last_trigger_time = now
    t = threading.Thread(target=_handle_spike, daemon=True)
    t.start()


# ----------------------------
# Main
# ----------------------------
def main():
    print("[AUDIO] Starting audio trigger listener")

    # Start arm state poller in background
    poller = threading.Thread(target=_poll_arm_state, daemon=True)
    poller.start()

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
        print("[AUDIO] Mic stream open.")
        print("[AUDIO] Waiting for game to arm via Ready button...")

        while True:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                if not _armed.is_set():
                    continue   # mic stays open but spikes are ignored when disarmed
                rms = rms_from_bytes(data)
                if DEBUG_RMS:
                    print(f"RMS={rms:.4f}")
                if rms >= RMS_THRESHOLD:
                    handle_spike_non_blocking()

            except IOError as e:
                print(f"[ERROR] Audio read: {e}")
                time.sleep(0.5)
            except Exception as e:
                print(f"[ERROR] Loop: {e}")
                time.sleep(0.5)
                continue

    except KeyboardInterrupt:
        print("\n[AUDIO] Stopped.")
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