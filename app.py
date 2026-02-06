from flask import Flask, render_template, request, jsonify
import subprocess
import os
import shutil
import cv2 # Moved cv2 import to the top as it's used in debug_overlay
import detect_dart  # uses your existing detection logic

app = Flask(__name__)

@app.errorhandler(Exception)
def handle_exception(e):
    print(f"UNHANDLED EXCEPTION: {e}")
    return jsonify({"ok": False, "error": "unhandled_exception", "details": str(e)}), 500

# Paths for before/after images used by detection
BEFORE_PATH = "before.jpg"
AFTER_PATH = "after.jpg"

@app.route('/')
def index():
    """Renders the default index page (could be a dashboard or static info)."""
    return render_template('index.html')

@app.post('/hit')
def hit():
    """Receives and logs hit information from the frontend."""
    data = request.get_json(force=True)
    print(f"WEBHOOK HIT: {data}") # shows in your terminal
    return jsonify({"ok": True})

@app.route('/home')
def home():
    """Renders the home page of the application."""
    return render_template('home.html')

@app.route('/play/<game>')
def play(game):
    """
    Renders the game page, initializing game parameters based on URL query.
    Supports 'around' and 'x01' game modes.
    """
    # Restrict to known games, default to 'around'
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
    Captures a 'BEFORE' frame (no dart) and saves it to BEFORE_PATH.
    This serves as the baseline image for dart detection.
    """
    try:
        cmd = [
            "rpicam-still",
            "-o", BEFORE_PATH,
            "-t", "1000", # Shutter time in milliseconds
            "--width", "1920",
            "--height", "1080",
            # Consider adding --nopreview if you don't need a preview window
        ]
        print(f"[CAPTURE] Executing command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=5)
        print(f"[CAPTURE] Successfully captured {BEFORE_PATH}.")
        return jsonify({"ok": True})
    except subprocess.CalledProcessError as e:
        print(f"ERROR capturing BEFORE image: {e}")
        # Log stderr for more detailed error from rpicam-still
        if e.stderr:
            print(f"rpicam-still stderr: {e.stderr}")
        return jsonify({"ok": False, "error": "capture_before_failed", "details": str(e)}), 500
    except Exception as e:
        print(f"UNEXPECTED ERROR in capture_before: {e}")
        return jsonify({"ok": False, "error": "unexpected_capture_error", "details": str(e)}), 500

@app.post('/detect')
def detect():
    """
    Captures an 'AFTER' frame, compares it against the baseline,
    and returns detected dart hit information.
    """
    # 0) Ensure we have a baseline; if not, capture one and return a warm-up response.
    if not os.path.exists(BEFORE_PATH):
        print(f"[DETECT] {BEFORE_PATH} not found. Capturing initial baseline.")
        try:
            cmd = [
                "rpicam-still",
                "-o", BEFORE_PATH,
                "-t", "1000",
                "--width", "1920",
                "--height", "1080",
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=5)
            print("[DETECT] Baseline captured (warm-up). No dart to detect yet.")
            return jsonify({"ok": True, "hit": None, "reason": "baseline_captured"})
        except subprocess.CalledProcessError as e:
            print(f"ERROR capturing initial baseline image: {e}")
            if e.stderr:
                print(f"rpicam-still stderr: {e.stderr}")
            return jsonify({"ok": False, "error": "baseline_capture_failed", "details": str(e)}), 500
        except Exception as e:
            print(f"UNEXPECTED ERROR capturing initial baseline: {e}")
            return jsonify({"ok": False, "error": "unexpected_baseline_error", "details": str(e)}), 500

    # 1) Capture AFTER image
    try:
        cmd = [
            "rpicam-still",
            "-o", AFTER_PATH,
            "-t", "1000",
            "--width", "1920",
            "--height", "1080",
        ]
        print(f"[DETECT] Executing command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=5)
        print(f"[DETECT] Successfully captured {AFTER_PATH}.")
    except subprocess.CalledProcessError as e:
        print(f"ERROR capturing AFTER image: {e}")
        if e.stderr:
            print(f"rpicam-still stderr: {e.stderr}")
        return jsonify({"ok": False, "error": "capture_after_failed", "details": str(e)}), 500
    except Exception as e:
        print(f"UNEXPECTED ERROR in capturing after image: {e}")
        return jsonify({"ok": False, "error": "unexpected_capture_error", "details": str(e)}), 500

    # 2) Load images and run high-level detection
    try:
        before = detect_dart.load_image(BEFORE_PATH)
        after = detect_dart.load_image(AFTER_PATH)

        if before is None:
            raise ValueError(f"Before image not loaded from {BEFORE_PATH}")
        if after is None:
            raise ValueError(f"After image not loaded from {AFTER_PATH}")

    except Exception as e:
        print(f"ERROR loading images for detection: {e}")
        return jsonify({"ok": False, "error": "load_failed", "details": str(e)}), 500

    # Removed the specific "no_aruco_warp" check
    # as detect_dart.py now always falls back to DEFAULT_SRC_POINTS if ArUco fails.
    result = detect_dart.detect_impact(before, after)
    # --- BACKEND DEBUG LOGGING (authoritative, non-invasive) ---
    dbg = result.get("debug", {})
    warp_src = dbg.get("warp", "unknown")

    if result.get("hit") is None:
        print(f"[DETECT] no hit | reason={result.get('reason')} | warp={warp_src}")
    else:
        h = result["hit"]
        print(
            f"[DETECT] hit "
            f"sector={h.get('sector')} ring={h.get('ring')} "
            f"r_frac={h.get('r_frac', 0):.3f} "
            f"warp={warp_src}"
        )
    # ----------------------------------------------------------
    
    # Save debug overlay with hit circle
    hit_info = result.get("hit")
    if hit_info:
        # Debug pixel info printed first as it's direct from detect_dart
        print(f"[DEBUG] Raw hit pixels from detect_dart: px={hit_info.get('px')}, py={hit_info.get('py')}")
        
        # Ensure 'after' image was successfully loaded before drawing on it
        if after is not None:
            debug_overlay_path = "last_overlay_debug.jpg"
            debug_overlay = after.copy()
            # Check if 'px' and 'py' are valid numbers before using
            if isinstance(hit_info.get('px'), (int, float)) and isinstance(hit_info.get('py'), (int, float)):
                cv2.circle(debug_overlay, (int(hit_info['px']), int(hit_info['py'])), 10, (0, 0, 255), 2)
                cv2.imwrite(debug_overlay_path, debug_overlay)
                print(f"[DEBUG] Debug overlay written to {debug_overlay_path}")
            else:
                 print(f"WARNING: 'px' or 'py' in hit_info are not numbers. Cannot write debug overlay.")
        else:
             print(f"WARNING: 'after' image was None. Cannot write debug overlay.")


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
        print(f"WARNING: Failed to update baseline: {e}")

    ring = hit_info.get("ring")
    sector = hit_info.get("sector")
    cx = float(hit_info.get("px", 0.0))
    cy = float(hit_info.get("py", 0.0))

    px = cx
    py = cy

    print(f"[DEBUG] Sending PIXEL coords to frontend: px={px:.1f}, py={py:.1f}")

    # Normalise bull labels a bit for the front-end
    if ring == "inner_bull":
        hit_type = "inner_bull"
    elif ring == "outer_bull":
        hit_type = "outer_bull"
    elif ring == "miss" or sector is None: # `sector is None` implies it couldn't classify, so treat as miss
        hit_type = "miss"
    else:
        hit_type = ring  # "single", "double", "treble"

    final_payload = {
        "ok": True,
        "hit": {
            "type": hit_type,
            "sector": sector,
            "px": px,
            "py": py,
        }
    }
    print(f"[DEBUG] Final JSON payload: {final_payload}")
    return jsonify(final_payload)


if __name__ == '__main__':
    # It's good practice to set host='0.0.0.0' for RPi to be accessible from other devices
    app.run(host="0.0.0.0", port=5050, debug=False) # Keep debug=False for production use
