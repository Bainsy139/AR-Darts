from flask import Flask, render_template, request, jsonify
import os
import shutil
import threading
import detect_dart
import cv2
from picamera2 import Picamera2

app = Flask(__name__)

# Paths for before/after images used by detection
BEFORE_PATH = "before.jpg"
AFTER_PATH  = "after.jpg"

# Latest detection result — written by /detect, consumed by /latest-hit
_latest_hit = None

# ------------------------------
# Camera — single warm instance
# ------------------------------
_cam_lock = threading.Lock()
_cam = None

def get_camera():
    """Return the shared warm Picamera2 instance, starting it if needed."""
    global _cam
    if _cam is None:
        print("[CAM] Starting camera...")
        _cam = Picamera2()
        config = _cam.create_still_configuration(
            main={"size": (1920, 1080), "format": "BGR888"},
            controls={
                "ExposureTime": 32680,
                "AnalogueGain": 6.0,
                "AwbEnable": False,
                "ColourGains": (1.5, 1.5),
            }
        )
        _cam.configure(config)
        _cam.start()
        import time; time.sleep(2)  # let exposure settle once
        print("[CAM] Camera ready.")
    return _cam

def capture_frame(path: str):
    """Capture a single frame to disk using the warm camera."""
    with _cam_lock:
        cam = get_camera()
        cam.capture_file(path)

# Warm the camera on startup
try:
    get_camera()
except Exception as e:
    print(f"[CAM] Warning: could not pre-warm camera: {e}")


# ------------------------------
# Routes
# ------------------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.post('/hit')
def hit():
    data = request.get_json(force=True)
    print("HIT:", data)
    return jsonify({"ok": True})

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/play/<game>')
def play(game):
    if game not in ('around', 'x01'):
        game = 'around'
    start = int(request.args.get('start', 501))
    if 'doubleOut' in request.args:
        double_out = request.args.get('doubleOut', 'true').lower() == 'true'
    else:
        double_out = request.args.get('double_out', '1') == '1'
    return render_template('play.html', game=game, start=start, double_out=double_out)


# ------------------------------
# Camera + detection endpoints
# ------------------------------

@app.post('/capture-before')
def capture_before():
    """Capture a BEFORE frame (no dart) and save to BEFORE_PATH."""
    try:
        capture_frame(BEFORE_PATH)
        print("[CAPTURE] Before frame saved.")
        return jsonify({"ok": True})
    except Exception as e:
        print(f"[ERROR] capture-before failed: {e}")
        return jsonify({"ok": False, "error": "capture_before_failed"}), 500


@app.post('/detect')
def detect():
    """Capture AFTER frame, diff against baseline, return hit JSON."""

    # 0) No baseline yet — capture one and return warm-up response
    if not os.path.exists(BEFORE_PATH):
        try:
            capture_frame(BEFORE_PATH)
            print("[DETECT] Baseline captured (warm-up).")
            return jsonify({"ok": True, "hit": None, "reason": "baseline_captured"})
        except Exception as e:
            print(f"[ERROR] baseline capture failed: {e}")
            return jsonify({"ok": False, "error": "baseline_capture_failed"}), 500

    # 1) Capture AFTER frame
    try:
        capture_frame(AFTER_PATH)
    except Exception as e:
        print(f"[ERROR] after capture failed: {e}")
        return jsonify({"ok": False, "error": "capture_after_failed"}), 500

    # 2) Load and detect
    try:
        before = detect_dart.load_image(BEFORE_PATH)
        after  = detect_dart.load_image(AFTER_PATH)
    except Exception as e:
        print(f"[ERROR] loading images: {e}")
        return jsonify({"ok": False, "error": "load_failed"}), 500

    result   = detect_dart.detect_impact(before, after)
    hit_info = result.get("hit")
    reason   = result.get("reason")

    # Save debug overlay
    if hit_info and 'x' in hit_info and 'y' in hit_info:
        debug_overlay = after.copy()
        cv2.circle(debug_overlay, (int(hit_info['x']), int(hit_info['y'])), 10, (0, 0, 255), 2)
        cv2.imwrite("last_overlay_debug.jpg", debug_overlay)

    if not hit_info:
        print(f"[DETECT] No impact detected (reason={reason}).")
        return jsonify({"ok": True, "hit": None, "reason": reason or "no_impact"})

    # Valid hit — update rolling baseline
    try:
        shutil.copy2(AFTER_PATH, BEFORE_PATH)
        print("[DETECT] Baseline updated after valid hit.")
    except Exception as e:
        print(f"[WARNING] Failed to update baseline: {e}")

    ring   = hit_info.get("ring")
    sector = hit_info.get("sector")
    cx     = float(hit_info.get("x", 0.0))
    cy     = float(hit_info.get("y", 0.0))

    if ring == "inner_bull":
        hit_type = "inner_bull"
    elif ring == "outer_bull":
        hit_type = "outer_bull"
    elif ring == "miss" or sector is None:
        hit_type = "miss"
    else:
        hit_type = ring  # "single", "double", "treble"

    hit_payload = {
        "type":   hit_type,
        "sector": sector,
        "x":      cx,
        "y":      cy,
    }

    global _latest_hit
    _latest_hit = hit_payload

    return jsonify({"ok": True, "hit": hit_payload})


@app.get('/latest-hit')
def latest_hit():
    """Return most recent detection result and clear it."""
    global _latest_hit
    hit     = _latest_hit
    _latest_hit = None
    return jsonify({"ok": True, "hit": hit})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5050, debug=True)