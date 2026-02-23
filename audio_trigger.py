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
import sys # Import sys for graceful exit

# ----------------------------
# Configuration (tuned values)
# ----------------------------
AUDIO_DEVICE_INDEX = 2   # None = default input (USB mic already set as capture)
SAMPLE_RATE = 44100
CHUNK = 1024                # frames per buffer
RMS_THRESHOLD = 0.1       # trigger threshold
COOLDOWN_SEC = 0.4          # minimum time between triggers
SETTLE_SEC = 0.30           # wait after impact before detection (dart vibration settle)
RETRY_ON_NO_IMPACT = True   # one retry if vision says "no_impact"
RETRY_DELAY_SEC = 0.20      # extra wait before retry
DEBUG_RMS = True            # set True to print live RMS
SERVER_BASE = "http://127.0.0.1:5050"

CAPTURE_BEFORE_ENDPOINT = f"{SERVER_BASE}/capture-before"
DETECT_ENDPOINT = f"{SERVER_BASE}/detect"

# ----------------------------
# Internal state
# ----------------------------
last_trigger_time = 0.0
# Use a semaphore or condition variable if you want to limit concurrent handle_spike calls
# For now, `busy` and `lock` are good enough for single-threaded handling of the Flask calls.
busy_handling_spike = False # Renamed for clarity
spike_lock = threading.Lock() # Renamed for clarity

# ----------------------------
# Helpers
# ----------------------------
def rms_from_bytes(data: bytes) -> float:
    """Compute RMS from 16-bit PCM audio chunk."""
    count = len(data) // 2
    if count == 0:
        return 0.0
    # Use 'h' for signed short (16-bit)
    samples = struct.unpack(f"{count}h", data)
    square_sum = 0.0
    for s in samples:
        norm = s / 32768.0 # Normalize to -1.0 to 1.0
        square_sum += norm * norm
    return math.sqrt(square_sum / count)

def post(endpoint: str):
    """POST helper with basic error handling. Returns (ok, json_data_or_none)."""
    try:
        r = requests.post(endpoint, timeout=5)
        if r.status_code != 200:
            print(f"[WARN] {endpoint} -> HTTP {r.status_code}")
            # If not 200 OK, return False and None, don't try to parse JSON
            return False, None
        try:
            # Only return JSON data if successfully parsed
            return True, r.json()
        except requests.exceptions.JSONDecodeError:
            print(f"[WARN] {endpoint} -> Received non-JSON response with HTTP 200")
            return True, None # Still consider it "ok" as it's 200, but no JSON data
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] POST {endpoint}: {e}")
        return False, None

def _handle_spike_blocking_actions():
    """
    Actions that block should run in a separate thread to not block audio acquisition.
    This includes sleeps and network requests.
    """
    global busy_handling_spike, last_trigger_time
    # Use a lock to ensure only one spike handling sequence runs at a time
    with spike_lock:
        if busy_handling_spike:
            return
        busy_handling_spike = True

    print("[SPIKE] Dart impact detected (processing in background)...")

    try:
        # Let the dart settle before we take the "after" frame inside /detect
        time.sleep(SETTLE_SEC)

        ok, data = post(DETECT_ENDPOINT)

        # If we fired too early, /detect may respond with no_impact. Retry once after a short delay.
        if RETRY_ON_NO_IMPACT and ok and isinstance(data, dict):
            reason = None
            try:
                # Access reason from `hit` or top-level, depending on backend structure
                reason = data.get("reason") or (data.get("hit") or {}).get("reason")
            except Exception:
                reason = None # Ensure reason is None on any access error

            if reason == "no_impact":
                print("[SPIKE] No impact detected by vision, retrying after delay...")
                time.sleep(RETRY_DELAY_SEC)
                post(DETECT_ENDPOINT) # No need to capture response for retry, just send

        # Refresh baseline for the next dart (this is usually done on successful hit)
        # If /detect is meant to refresh it, this call might be redundant or misplaced.
        # For simplicity, keeping it here as per original logic.
        print("[SPIKE] Refreshing baseline for next dart...")
        post(CAPTURE_BEFORE_ENDPOINT)

    except Exception as e:
        print(f"[ERROR] Error during spike handling: {e}")
    finally:
        with spike_lock: # Release the lock when done
            busy_handling_spike = False
        print("[SPIKE] Spike handling complete.")


def handle_spike_non_blocking():
    """
    Initiates the blocking spike handling actions in a separate thread.
    This function should be called from the main audio loop.
    """
    # Check cooldown within the main audio thread's execution for accuracy
    global last_trigger_time
    now = time.time()
    if (now - last_trigger_time) < COOLDOWN_SEC:
        return # Still in cooldown

    # Check if a previous spike is still being handled to prevent queueing
    with spike_lock:
        if busy_handling_spike:
            return # A request is already in flight

    last_trigger_time = now # Update last trigger time immediately
    
    # Start the actual blocking actions in a new thread
    spike_thread = threading.Thread(target=_handle_spike_blocking_actions)
    spike_thread.daemon = True # Allow main program to exit even if thread is running
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

        # Added error handling for opening stream
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=AUDIO_DEVICE_INDEX,
            frames_per_buffer=CHUNK,
            # No need for stream_callback as we are block-reading in a loop
        )

        print("[AUDIO] Mic stream open, listening for spikes")

        # Initial capture-before at game start
        print("[INIT] Capturing baseline frame...")
        # Since this happens only once, in the main thread is fine.
        ok, _ = post(CAPTURE_BEFORE_ENDPOINT)
        if not ok:
            print("[ERROR] Initial baseline capture failed. Check Flask server.")
            # Decide if you want to exit here or just warn. Exiting is safer.
            sys.exit(1)

        while True:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                rms = rms_from_bytes(data)
                if DEBUG_RMS:
                    print(f"RMS={rms:.4f}")

                if rms >= RMS_THRESHOLD:
                    handle_spike_non_blocking() # Call the non-blocking runner

            except IOError as e:
                # This could be buffer overflow or device unplugged
                print(f"[ERROR] Audio stream read error: {e}")
                # Optional: attempt to re-open stream or exit
                time.sleep(1) # Prevent busy-looping on error
            except Exception as e:
                print(f"[ERROR] An unexpected error occurred in audio loop: {e}")
                break # Exit loop on unexpected error

    except pyaudio.PyAudioError as e:
        print(f"[CRITICAL ERROR] PyAudio failed to initialize or open stream: {e}")
        print("Please check your microphone connection and configuration.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[AUDIO] Stopping listener due to KeyboardInterrupt.")
    except Exception as e:
        print(f"[CRITICAL ERROR] Unexpected error in main function: {e}")
    finally:
        print("[AUDIO] Cleaning up audio resources...")
        if stream and stream.is_active():
            stream.stop_stream()
        if stream:
            stream.close()
        if pa:
            pa.terminate()
        # Optional: wait for any background spike handling to complete
        # This might block exit for a few seconds if a request is in flight
        # print("[AUDIO] Waiting for pending spike handling to finish...")
        # with spike_lock: # This ensures _handle_spike_blocking_actions completes its critical section
        #    pass
        print("[AUDIO] Audio resources cleaned up.")

if __name__ == "__main__":
    main()
