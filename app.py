from flask import Flask, render_template, request, jsonify
import subprocess
import os
import shutil
import detect_dart  # uses your existing detection logic
import threading
import time
from enum import Enum, auto
import cv2
import numpy as np

app = Flask(__name__)

class GameState(Enum):
    HOME = auto()
    GAME_INIT = auto()
    ARMED = auto()
    THROW_IN_PROGRESS = auto()
    LOCK = auto()
    TURN_CHANGE = auto()
    WIN = auto()

STATE = GameState.HOME
STATE_LOCK = threading.Lock()

current_player = 0
darts_thrown = 0
MAX_DARTS = 3

# Fake motion signal for now (0.0 = still, 1.0 = motion)
motion_level = 0.0
motion_stable_frames = 0
MOTION_THRESHOLD = 1.4
STABLE_FRAMES_REQUIRED = 3

# --- Motion detection (simple frame-diff) ---
CAM_WIDTH = 640
CAM_HEIGHT = 360
CAM_CMD = [
    "rpicam-still",
    "--nopreview",
    "-t", "1",
    "--width", str(CAM_WIDTH),
    "--height", str(CAM_HEIGHT),
    "-o", "-"
]

_prev_gray = None

def capture_gray_frame():
    """
    Capture a low-res frame and return grayscale numpy array.
    Uses rpicam-still stdout pipe for minimal latency.
    On non-Pi systems, returns None.
    """
    try:
        p = subprocess.Popen(CAM_CMD, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        img_bytes = p.stdout.read()
        p.wait(timeout=1)
        if not img_bytes:
            return None
        img = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
        return frame
    except Exception:
        return None

def compute_motion_level(frame_gray):
    """
    Compute a single scalar motion energy from frame diff.
    """
    global _prev_gray
    if frame_gray is None:
        return 0.0
    if _prev_gray is None:
        _prev_gray = frame_gray
        return 0.0
    diff = cv2.absdiff(_prev_gray, frame_gray)
    _prev_gray = frame_gray
    # Mean absolute difference as motion energy
    return float(np.mean(diff))

def set_state(new_state):
    global STATE
    with STATE_LOCK:
        print(f"[STATE] {STATE.name} → {new_state.name}")
        STATE = new_state

# Paths for before/after images used by detection
BEFORE_PATH = "before.jpg"
AFTER_PATH = "after.jpg"


@app.route('/')
def index():
    return render_template('index.html')


@app.post('/hit')
def hit():
    data = request.get_json(force=True)
    print("HIT:", data)  # shows in your terminal
    return jsonify({"ok": True})


@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/play/<game>')
def play(game):
    # restrict to known games
    if game not in ('around', 'x01'):
        game = 'around'
    start = int(request.args.get('start', 501))
    # Support both legacy double_out=1 and newer doubleOut=true
    if 'doubleOut' in request.args:
        double_out = request.args.get('doubleOut', 'true').lower() == 'true'
    else:
        double_out = request.args.get('double_out', '1') == '1'
    return render_template('play.html', game=game, start=start, double_out=double_out)


# --- New endpoints for camera + detection --- #

@app.post('/capture-before')
def capture_before():
    """
    Capture a BEFORE frame (no dart) and save to BEFORE_PATH.
    Front-end flow:
      1) Call /capture-before
      2) User throws dart
      3) Call /detect
    """
    try:
        cmd = [
            "rpicam-still",
            "-o", BEFORE_PATH,
            "-t", "1000",
            "--width", "1920",
            "--height", "1080",
        ]
        subprocess.run(cmd, check=True)
        return jsonify({"ok": True})
    except subprocess.CalledProcessError as e:
        print("ERROR capturing BEFORE image:", e)
        return jsonify({"ok": False, "error": "capture_before_failed"}), 500


@app.post('/detect')
def detect():
    """Capture an AFTER frame, compare against a rolling baseline, and return hit JSON.

    Behaviour:
      - If BEFORE_PATH (baseline) does not exist yet, capture a warm-up frame
        and return a no_impact response. This establishes the baseline.
      - On subsequent calls, use BEFORE_PATH as the baseline, capture AFTER_PATH
        as the new frame, and run detect_dart.detect_impact(before, after).
      - When a valid hit is found, update BEFORE_PATH to the latest AFTER_PATH
        so multiple darts can accumulate on the board.
    """
    # 0) Ensure we have a baseline; if not, capture one and return a warm-up response.
    if not os.path.exists(BEFORE_PATH):
        try:
            cmd = [
                "rpicam-still",
                "-o", BEFORE_PATH,
                "-t", "1000",
                "--width", "1920",
                "--height", "1080",
            ]
            subprocess.run(cmd, check=True)
            print("[DETECT] Baseline captured (warm-up).")
            return jsonify({"ok": True, "hit": None, "reason": "baseline_captured"})
        except subprocess.CalledProcessError as e:
            print("ERROR capturing baseline image:", e)
            return jsonify({"ok": False, "error": "baseline_capture_failed"}), 500

    # 1) Capture AFTER image
    try:
        cmd = [
            "rpicam-still",
            "-o", AFTER_PATH,
            "-t", "1000",
            "--width", "1920",
            "--height", "1080",
        ]
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print("ERROR capturing AFTER image:", e)
        return jsonify({"ok": False, "error": "capture_after_failed"}), 500

    # 2) Load images and run high-level detection
    try:
        before = detect_dart.load_image(BEFORE_PATH)
        after = detect_dart.load_image(AFTER_PATH)
    except Exception as e:
        print("ERROR loading images:", e)
        return jsonify({"ok": False, "error": "load_failed"}), 500

    result = detect_dart.detect_impact(before, after)
    # Save debug overlay with hit circle
    hit_info = result.get("hit")
    if hit_info and 'x' in hit_info and 'y' in hit_info:
        debug_overlay = after.copy()
        import cv2
        cv2.circle(debug_overlay, (int(hit_info['x']), int(hit_info['y'])), 10, (0, 0, 255), 2)
        cv2.imwrite("last_overlay_debug.jpg", debug_overlay)

    reason = result.get("reason")

    if not hit_info:
        # No clear impact or off-board; keep the existing baseline for now.
        print(f"[DETECT] No impact detected (reason={reason}).")
        return jsonify({"ok": True, "hit": None, "reason": reason or "no_impact"})

    # We have a valid hit; update the rolling baseline to the latest frame
    try:
        shutil.copy2(AFTER_PATH, BEFORE_PATH)
        print("[DETECT] Baseline updated after valid hit.")
    except Exception as e:
        print("WARNING: Failed to update baseline:", e)

    ring = hit_info.get("ring")
    sector = hit_info.get("sector")
    cx = float(hit_info.get("x", 0.0))
    cy = float(hit_info.get("y", 0.0))

    # Normalise bull labels a bit for the front-end
    if ring == "inner_bull":
        hit_type = "inner_bull"
    elif ring == "outer_bull":
        hit_type = "outer_bull"
    elif ring == "miss" or sector is None:
        hit_type = "miss"
    else:
        hit_type = ring  # "single", "double", "treble"

    return jsonify({
        "ok": True,
        "hit": {
            "type": hit_type,
            "sector": sector,
            "x": cx,
            "y": cy,
        }
    })

@app.post('/debug/start-game')
def debug_start_game():
    set_state(GameState.GAME_INIT)
    return jsonify({"ok": True, "state": STATE.name})

def game_loop():
    global STATE, motion_level, motion_stable_frames, darts_thrown, current_player

    print("[GAME LOOP] Started")
    tick_rate = 0.1  # seconds (10 Hz)

    while True:
        time.sleep(tick_rate)

        # Update motion level from camera
        frame_gray = capture_gray_frame()
        motion_level = compute_motion_level(frame_gray)
        print(f"[MOTION] {motion_level:.2f}")

        with STATE_LOCK:
            state = STATE

        # ---- HOME ----
        if state == GameState.HOME:
            continue

        # ---- GAME INIT ----
        if state == GameState.GAME_INIT:
            current_player = 0
            darts_thrown = 0
            motion_stable_frames = 0
            set_state(GameState.ARMED)
            continue

        # ---- ARMED ----
        if state == GameState.ARMED:
            if motion_level > MOTION_THRESHOLD:
                motion_stable_frames = 0
                set_state(GameState.THROW_IN_PROGRESS)
            continue

        # ---- THROW IN PROGRESS ----
        if state == GameState.THROW_IN_PROGRESS:
            if motion_level <= MOTION_THRESHOLD:
                motion_stable_frames += 1
                if motion_stable_frames >= STABLE_FRAMES_REQUIRED:
                    set_state(GameState.LOCK)
            else:
                motion_stable_frames = 0
            continue

        # ---- LOCK ----
        if state == GameState.LOCK:
            darts_thrown += 1
            print(f"[LOCK] Dart detected for Player {current_player + 1} (dart {darts_thrown})")

            # Stub: no detection yet, just simulate a valid hit
            if darts_thrown >= MAX_DARTS:
                set_state(GameState.TURN_CHANGE)
            else:
                set_state(GameState.ARMED)
            continue

        # ---- TURN CHANGE ----
        if state == GameState.TURN_CHANGE:
            current_player = (current_player + 1) % 2
            darts_thrown = 0
            print(f"[TURN] Now Player {current_player + 1}")
            set_state(GameState.ARMED)
            continue

        # ---- WIN ----
        if state == GameState.WIN:
            print("[WIN] Game over, returning to HOME")
            set_state(GameState.HOME)
            continue

if __name__ == '__main__':
    threading.Thread(target=game_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=5050, debug=True)