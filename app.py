from flask import Flask, render_template, request, jsonify
import subprocess
import detect_dart  # uses your existing detection logic

app = Flask(__name__)

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
    """
    Capture an AFTER frame (with dart), run detection against BEFORE_PATH,
    and return the classified hit as JSON.
    """
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

    # 2) Run detection using your existing detect_dart.py functions
    try:
        before = detect_dart.load_image(BEFORE_PATH)
        after = detect_dart.load_image(AFTER_PATH)
    except Exception as e:
        print("ERROR loading images:", e)
        return jsonify({"ok": False, "error": "load_failed"}), 500

    center, _mask = detect_dart.find_dart_center(before, after)

    if center is None:
        # No clear blob – treat as miss
        return jsonify({"ok": True, "hit": None, "reason": "no_impact"})

    cx, cy = center
    ring, sector = detect_dart.classify_hit(cx, cy)

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
            "type": hit_type,   # "single" | "double" | "treble" | "inner_bull" | "outer_bull" | "miss"
            "sector": sector,   # 1–20 or None for bulls/miss
            "x": cx,
            "y": cy,
        }
    })


if __name__ == '__main__':
    app.run(debug=True)