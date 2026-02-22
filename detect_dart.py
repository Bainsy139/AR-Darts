import sys
import math
import cv2
import numpy as np

# Try to import ArUco for optional marker-based calibration
try:
    import cv2.aruco as aruco
    HAS_ARUCO = True
except ImportError:
    HAS_ARUCO = False

# ------------------------------
# CONFIG
# ------------------------------

# Board position in the flat (warped) image.
# If warp succeeds, these describe the board in the corrected frame.
# If warp fails, these describe the board in the raw flipped image.
BOARD_CX = 993
BOARD_CY = 344
BOARD_RADIUS = 206

ROT_OFFSET_DEG = -8.0

SECTORS = [
    20, 1, 18, 4, 13, 6, 10, 15, 2, 17,
    3, 19, 7, 16, 8, 11, 14, 9, 12, 5
]

CAMERA_UPSIDE_DOWN = True

# Set to True once ArUco warp is confirmed working.
# Set to False to use raw flipped image (current working state).
USE_ARUCO_WARP = True

# Output frame size
FRAME_W = 1920
FRAME_H = 669

# ArUco warp destination: the 4 markers map to the full frame corners.
# Order matches marker IDs: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
ARUCO_DST = np.float32([
    [0,    0  ],   # id=0 → top-left
    [1920, 0  ],   # id=1 → top-right
    [0,    669],   # id=2 → bottom-left
    [1920, 669],   # id=3 → bottom-right
])

# detection tuning
DIFF_THRESHOLD = 15
MIN_BLOB_AREA = 10


# ------------------------------
# RING / SECTOR CLASSIFICATION
# ------------------------------

def ring_from_radius_frac(r_frac: float) -> str:
    if r_frac <= 0.035:
        return "inner_bull"
    if r_frac <= 0.09:
        return "outer_bull"
    if r_frac < 0.57:
        return "single"
    if r_frac <= 0.63:
        return "treble"
    if r_frac < 0.95:
        return "single"
    if r_frac <= 1.0:
        return "double"
    return "miss"

def sector_index_from_angle(angle: float) -> int:
    rot_rad = math.radians(ROT_OFFSET_DEG)
    a = angle + math.pi / 2 - rot_rad
    a = (a % (2 * math.pi) + 2 * math.pi) % (2 * math.pi)
    return int(math.floor(a / (2 * math.pi) * 20)) % 20

def pixel_to_polar(x: float, y: float):
    dx = x - BOARD_CX
    dy = y - BOARD_CY
    r = math.hypot(dx, dy)
    angle = math.atan2(dy, dx)
    return r / max(1.0, BOARD_RADIUS), angle

def classify_hit_with_debug(x: float, y: float):
    r_frac, angle = pixel_to_polar(x, y)
    angle_deg = math.degrees(angle) % 360
    ring = ring_from_radius_frac(r_frac)
    sector = None if ring == "miss" else SECTORS[sector_index_from_angle(angle)]
    print(f"[DEBUG] Tip at ({x:.1f},{y:.1f}) => angle {angle_deg:.1f}°, sector {sector}")
    return {"ring": ring, "sector": sector, "r_frac": r_frac, "angle_deg": angle_deg}

def classify_hit(x, y):
    info = classify_hit_with_debug(x, y)
    print(f"[HIT] Estimated sector={info['sector']}, type={info['ring']}")
    return info["ring"], info["sector"]


# ------------------------------
# ARUCO WARP
# ------------------------------

def _compute_warp_from_aruco(img):
    """
    Detects 4 ArUco markers (IDs 0-3) and computes a perspective warp
    that maps the surface defined by those markers into a flat full-frame rectangle.
    Returns the warp matrix M, or None if detection fails.
    """
    if not HAS_ARUCO:
        print("[ARUCO] cv2.aruco not available.")
        return None

    try:
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        params = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(dictionary, params)
    except Exception as e:
        print(f"[ARUCO] Failed to create detector: {e}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        print("[ARUCO] No markers detected. Using raw image.")
        return None

    ids_flat = ids.flatten()
    required = [0, 1, 2, 3]

    if not all(r in ids_flat for r in required):
        print(f"[ARUCO] Only found markers: {ids_flat}. Need all of [0,1,2,3]. Using raw image.")
        return None

    # Build source points in marker ID order [0,1,2,3]
    src = []
    for mid in required:
        idx = int(np.where(ids_flat == mid)[0][0])
        centre = corners[idx][0].mean(axis=0)
        src.append(centre)

    src = np.float32(src)
    M = cv2.getPerspectiveTransform(src, ARUCO_DST)
    print("[ARUCO] All 4 markers found. Warp matrix computed.")
    return M


# ------------------------------
# IMAGE LOADING
# ------------------------------

def load_image(path: str):
    """
    Loads an image, flips it if camera is upside down, and optionally
    applies ArUco-based perspective correction.
    """
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Failed to load {path}")

    if CAMERA_UPSIDE_DOWN:
        img = cv2.rotate(img, cv2.ROTATE_180)

    if USE_ARUCO_WARP:
        M = _compute_warp_from_aruco(img)
        if M is not None:
            img = cv2.warpPerspective(img, M, (FRAME_W, FRAME_H))
        else:
            print("[WARP] Falling back to raw flipped image.")

    return img


# ------------------------------
# DART DETECTION
# ------------------------------

def preprocess_for_diff(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def find_dart_center(before_img, after_img, debug_img=None):
    g_before = preprocess_for_diff(before_img)
    g_after = preprocess_for_diff(after_img)

    diff = cv2.absdiff(g_before, g_after)

    h, w = diff.shape
    yy, xx = np.mgrid[0:h, 0:w]
    dx = xx - BOARD_CX
    dy = yy - BOARD_CY
    r = np.sqrt(dx * dx + dy * dy)
    mask = (r <= BOARD_RADIUS * 1.1)
    diff = np.where(mask, diff, 0).astype(np.uint8)

    diff_bin = (diff > DIFF_THRESHOLD).astype(np.uint8) * 255

    k = np.ones((3, 3), np.uint8)
    diff_bin = cv2.morphologyEx(diff_bin, cv2.MORPH_CLOSE, k, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(diff_bin, connectivity=8)

    best_label = None
    best_top = None
    best_area = 0

    for lab in range(1, num_labels):
        area = stats[lab, cv2.CC_STAT_AREA]
        if area < MIN_BLOB_AREA:
            continue
        top = stats[lab, cv2.CC_STAT_TOP]
        if best_label is None or top < best_top or (top == best_top and area > best_area):
            best_label = lab
            best_top = top
            best_area = area

    if best_label is None:
        if debug_img is not None:
            cv2.imwrite("debug_last_blob.jpg", debug_img)
        return None, diff_bin

    comp = (labels == best_label).astype(np.uint8) * 255

    if debug_img is not None:
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            cv2.drawContours(debug_img, [c], -1, (0, 255, 0), 2)

    ys, xs = np.where(comp > 0)
    if len(xs) == 0:
        if debug_img is not None:
            cv2.imwrite("debug_last_blob.jpg", debug_img)
        return None, comp

    i = np.argmin(ys)
    tip = (float(xs[i]), float(ys[i]))

    if debug_img is not None:
        cv2.circle(debug_img, (int(tip[0]), int(tip[1])), 4, (0, 0, 255), -1)

    return tip, comp


# ------------------------------
# DEBUG OVERLAY DRAWING
# ------------------------------

def _draw_overlay(img):
    """Draws board rings and sector lines onto img in-place."""
    center = (int(BOARD_CX), int(BOARD_CY))

    # Outer ring
    cv2.circle(img, center, int(BOARD_RADIUS), (0, 0, 255), 2)

    # Inner rings
    for frac in [0.035, 0.09, 0.57, 0.63, 0.95]:
        cv2.circle(img, center, int(BOARD_RADIUS * frac), (0, 255, 0), 1)

    # Sector lines with labels
    rot_rad = math.radians(ROT_OFFSET_DEG)
    for k in range(20):
        angle = -math.pi / 2 + rot_rad + k * (2 * math.pi / 20)
        x2 = int(BOARD_CX + BOARD_RADIUS * math.cos(angle))
        y2 = int(BOARD_CY + BOARD_RADIUS * math.sin(angle))
        cv2.line(img, center, (x2, y2), (255, 0, 0), 1)

        lx = int(BOARD_CX + BOARD_RADIUS * 1.05 * math.cos(angle))
        ly = int(BOARD_CY + BOARD_RADIUS * 1.05 * math.sin(angle))
        cv2.putText(img, str(SECTORS[k]), (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

    cv2.circle(img, center, 3, (255, 255, 255), -1)

def draw_debug_overlay(input_path: str, output_path: str):
    img = load_image(input_path)
    _draw_overlay(img)
    cv2.imwrite(output_path, img)

def draw_debug_overlay_with_hit(input_path: str, hit_xy, output_path: str):
    img = load_image(input_path)
    _draw_overlay(img)

    if hit_xy is not None:
        pt = (int(hit_xy[0]), int(hit_xy[1]))
        cv2.circle(img, pt, 12, (255, 255, 255), 2)
        cv2.circle(img, pt, 10, (0, 0, 255), -1)
        cv2.circle(img, pt, 2, (0, 0, 0), -1)

    cv2.imwrite(output_path, img)


# ------------------------------
# MAIN
# ------------------------------

def main():

    # MODE: overlay
    if len(sys.argv) >= 2 and sys.argv[1] == "overlay":
        if len(sys.argv) != 4:
            print("Usage: python3 detect_dart.py overlay INPUT.jpg OUTPUT.jpg")
            sys.exit(1)
        draw_debug_overlay(sys.argv[2], sys.argv[3])
        print(f"Overlay written to {sys.argv[3]}")
        sys.exit(0)

    # MODE: overlayhit
    if len(sys.argv) >= 2 and sys.argv[1] == "overlayhit":
        if len(sys.argv) != 5:
            print("Usage: python3 detect_dart.py overlayhit BEFORE.jpg AFTER.jpg OUTPUT.jpg")
            sys.exit(1)
        import datetime
        before = load_image(sys.argv[2])
        after = load_image(sys.argv[3])
        debug = after.copy()
        hit_point, _ = find_dart_center(before, after, debug)
        print(f"[DEBUG] Estimated tip @ {hit_point}")
        draw_debug_overlay_with_hit(sys.argv[3], hit_point, sys.argv[4])
        print(f"Overlay+hit written to {sys.argv[4]}")
        sys.exit(0)

    # MODE: aruco (diagnostic - shows which markers are detected)
    if len(sys.argv) >= 2 and sys.argv[1] == "aruco":
        if len(sys.argv) != 4:
            print("Usage: python3 detect_dart.py aruco INPUT.jpg OUTPUT.jpg")
            sys.exit(1)
        img = load_image(sys.argv[2])
        try:
            dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
            params = aruco.DetectorParameters()
            detector = aruco.ArucoDetector(dictionary, params)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is not None:
                aruco.drawDetectedMarkers(img, corners, ids)
                print(f"[ARUCO] Found markers: {ids.flatten()}")
            else:
                print("[ARUCO] No markers detected.")
        except Exception as e:
            print(f"[ARUCO] Error: {e}")
        cv2.imwrite(sys.argv[3], img)
        print(f"Saved ArUco debug to {sys.argv[3]}")
        sys.exit(0)

    # Default: detect impact
    if len(sys.argv) != 3:
        print("Usage:")
        print("  python3 detect_dart.py BEFORE.jpg AFTER.jpg")
        print("  python3 detect_dart.py overlay INPUT.jpg OUTPUT.jpg")
        print("  python3 detect_dart.py overlayhit BEFORE.jpg AFTER.jpg OUTPUT.jpg")
        print("  python3 detect_dart.py aruco INPUT.jpg OUTPUT.jpg")
        sys.exit(1)

    before = load_image(sys.argv[1])
    after = load_image(sys.argv[2])
    tip, _ = find_dart_center(before, after)

    if tip is None:
        print("No clear dart impact detected.")
        sys.exit(0)

    cx, cy = tip
    ring, sector = classify_hit(cx, cy)

    if ring == "miss" or sector is None:
        print(f"Detected ({cx:.1f},{cy:.1f}) but MISS")
    else:
        print(f"Detected impact: {ring} {sector} at ({cx:.1f},{cy:.1f})")


# ------------------------------
# APP HELPERS
# ------------------------------

def detect_impact(before_img, after_img):
    """Called by app.py when camera is live."""
    tip, _ = find_dart_center(before_img, after_img)
    if tip is None:
        return {"hit": None, "reason": "no_impact"}
    x, y = tip
    info = classify_hit_with_debug(x, y)
    info["x"] = x
    info["y"] = y
    if info["ring"] == "miss" or info["sector"] is None:
        return {"hit": None, "reason": "off_board"}
    return {"hit": info, "reason": None}

def estimate_tip(before_img, after_img, debug_img=None):
    """Returns estimated dart tip position."""
    tip, _ = find_dart_center(before_img, after_img, debug_img)
    if tip is None:
        print("[ERROR] No dart pixels found.")
        return None
    print(f"[DEBUG] Tip estimate: ({tip[0]:.1f}, {tip[1]:.1f})")
    return tip


if __name__ == "__main__":
    main()
