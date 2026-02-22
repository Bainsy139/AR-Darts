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

BOARD_CX = 930
BOARD_CY = 625
BOARD_RADIUS = 130

ANGLE_OFFSET_DEGREES = 0
ROT_OFFSET_DEG = -9.8

SECTORS = [
    20, 1, 18, 4, 13, 6, 10, 15, 2, 17,
    3, 19, 7, 16, 8, 11, 14, 9, 12, 5
]

CAMERA_UPSIDE_DOWN = True

USE_WARP = True
USE_ARUCO_WARP = True

DEFAULT_SRC_POINTS = np.float32([
    [1039, 483],  # top
    [1195, 617],  # right
    [1045, 738],  # bottom
    [890,  627],  # left
])

DST_POINTS = np.float32([
    [BOARD_CX,               BOARD_CY - BOARD_RADIUS],  # top
    [BOARD_CX + BOARD_RADIUS, BOARD_CY],                # right
    [BOARD_CX,               BOARD_CY + BOARD_RADIUS],  # bottom
    [BOARD_CX - BOARD_RADIUS, BOARD_CY],                # left
])

WARP_MATRIX = None

# detection tuning
DIFF_THRESHOLD = 15
MIN_BLOB_AREA = 10

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
    a = (a % (2*math.pi) + 2*math.pi) % (2*math.pi)
    return int(math.floor(a / (2*math.pi) * 20)) % 20

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

    if ring == "miss":
        sector = None
    else:
        idx = sector_index_from_angle(angle)
        sector = SECTORS[idx]

    print(f"[DEBUG] Tip at ({x:.1f},{y:.1f}) => angle {angle_deg:.1f}°, sector {sector}")
    return {
        "ring": ring,
        "sector": sector,
        "r_frac": r_frac,
        "angle_deg": angle_deg
    }

def classify_hit(x, y):
    info = classify_hit_with_debug(x, y)
    print(f"[HIT] Estimated sector={info['sector']}, type={info['ring']}")
    return info["ring"], info["sector"]

def _compute_warp_from_aruco(img):
    if not HAS_ARUCO or not USE_ARUCO_WARP:
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
    
    if ids is None or len(ids) < 4:
        print(f"[ARUCO] Detection failed - found {0 if ids is None else len(ids)} markers, need 4. Falling back to DEFAULT_SRC_POINTS")
        return None

    ids = ids.flatten()
    required = [0, 1, 2, 3]
    if not all(r in ids for r in required):
        print(f"[ARUCO] Missing required markers. Found: {ids}. Falling back to DEFAULT_SRC_POINTS")
        return None

    src = []
    for mid in required:
        idx = int(np.where(ids == mid)[0][0])
        c = corners[idx][0].mean(axis=0)
        src.append(c)

    print(f"[ARUCO] All 4 markers found! Computing warp matrix.")
    src = np.float32(src)
    return cv2.getPerspectiveTransform(src, DST_POINTS)

def load_image(path: str):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Failed to load {path}")

    if CAMERA_UPSIDE_DOWN:
        img = cv2.rotate(img, cv2.ROTATE_180)

    if USE_WARP:
        global WARP_MATRIX
        h, w = img.shape[:2]

        if WARP_MATRIX is None:
            M = _compute_warp_from_aruco(img)
            if M is None:
                M = cv2.getPerspectiveTransform(DEFAULT_SRC_POINTS, DST_POINTS)
            WARP_MATRIX = M

        img = cv2.warpPerspective(img, WARP_MATRIX, (w, h))

    return img

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
    r = np.sqrt(dx*dx + dy*dy)
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

        if best_label is None:
            best_label = lab
            best_top = top
            best_area = area
            continue

        if top < best_top or (top == best_top and area > best_area):
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
            cv2.drawContours(debug_img, [c], -1, (0,255,0), 2)

    ys, xs = np.where(comp > 0)
    if len(xs) == 0:
        if debug_img is not None:
            cv2.imwrite("debug_last_blob.jpg", debug_img)
        return None, comp

    i = np.argmin(ys)
    tip = (float(xs[i]), float(ys[i]))

    if debug_img is not None:
        cv2.circle(debug_img, (int(tip[0]), int(tip[1])), 4, (0,0,255), -1)

    return tip, comp


def draw_debug_overlay(input_path: str, output_path: str):
    img = load_image(input_path)
    overlay = img.copy()

    center = (int(BOARD_CX), int(BOARD_CY))

    cv2.circle(overlay, center, int(BOARD_RADIUS), (0,0,255), 2)

    ring_fracs = [0.035, 0.09, 0.57, 0.63, 0.95]
    for frac in ring_fracs:
        r = int(BOARD_RADIUS * frac)
        cv2.circle(overlay, center, r, (0,255,0), 1)

    rot_rad = math.radians(ROT_OFFSET_DEG)
    two_pi = 2*math.pi
    for k in range(20):
        angle = -math.pi/2 + rot_rad + k*(two_pi/20)
        x2 = int(BOARD_CX + BOARD_RADIUS * math.cos(angle))
        y2 = int(BOARD_CY + BOARD_RADIUS * math.sin(angle))
        cv2.line(overlay, center, (x2, y2), (255,0,0), 1)

    cv2.circle(overlay, center, 3, (255,255,255), -1)

    cv2.imwrite(output_path, overlay)


def draw_debug_overlay_with_hit(input_path: str, hit_xy, output_path: str):
    img = load_image(input_path)
    overlay = img.copy()

    center = (int(BOARD_CX), int(BOARD_CY))

    cv2.circle(overlay, center, int(BOARD_RADIUS), (0,0,255), 2)

    ring_fracs = [0.035, 0.09, 0.57, 0.63, 0.95]
    for frac in ring_fracs:
        r = int(BOARD_RADIUS * frac)
        cv2.circle(overlay, center, r, (0,255,0), 1)

    cv2.circle(overlay, center, int(BOARD_RADIUS * 0.035), (0,255,255), 2)
    cv2.circle(overlay, center, int(BOARD_RADIUS * 0.09), (0,128,255), 1)

    rot_rad = math.radians(ROT_OFFSET_DEG)
    two_pi = 2*math.pi
    for k in range(20):
        angle = -math.pi/2 + rot_rad + k*(two_pi/20)
        x2 = int(BOARD_CX + BOARD_RADIUS * math.cos(angle))
        y2 = int(BOARD_CY + BOARD_RADIUS * math.sin(angle))
        cv2.line(overlay, center, (x2, y2), (255,0,0), 1)

        label_r = int(BOARD_RADIUS * 1.05)
        lx = int(BOARD_CX + label_r * math.cos(angle))
        ly = int(BOARD_CY + label_r * math.sin(angle))
        cv2.putText(overlay, str(SECTORS[k]), (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1, cv2.LINE_AA)

    cv2.circle(overlay, center, 3, (255,255,255), -1)

    if hit_xy is not None:
        hx, hy = hit_xy
        pt = (int(hx), int(hy))
        cv2.circle(overlay, pt, 12, (255,255,255), 2)
        cv2.circle(overlay, pt, 10, (0,0,255), -1)
        cv2.circle(overlay, pt, 2, (0,0,0), -1)

    cv2.imwrite(output_path, overlay)

def main():
    # ---------------------------
    # MODE: overlay
    # ---------------------------
    if len(sys.argv) >= 2 and sys.argv[1] == "overlay":
        if len(sys.argv) != 4:
            print("Usage: python3 detect_dart.py overlay INPUT.jpg OUTPUT.jpg")
            sys.exit(1)

        input_path = sys.argv[2]
        out_path = sys.argv[3]
        draw_debug_overlay(input_path, out_path)
        print(f"Overlay written to {out_path}")
        sys.exit(0)

    # ---------------------------
    # MODE: overlayhit
    # ---------------------------
    if len(sys.argv) >= 2 and sys.argv[1] == "overlayhit":
        if len(sys.argv) != 5:
            print("Usage: python3 detect_dart.py overlayhit BEFORE.jpg AFTER.jpg OUTPUT.jpg")
            sys.exit(1)

        import datetime
        before_path = sys.argv[2]
        after_path = sys.argv[3]
        out_path = sys.argv[4]

        # ALL IMAGES WARPED ONCE HERE
        before = load_image(before_path)
        after = load_image(after_path)
        overlay = load_image(after_path)

        # Images already rotated + warped
        h, w = before.shape[:2]

        # Compute hit
        hit_point, edges = find_dart_center(before, after, overlay)
        print(f"DEBUG: Estimated tip @ {hit_point}")

        # Draw overlay
        after_img = after.copy()
        center = (int(BOARD_CX), int(BOARD_CY))

        if hit_point is not None:
            tx, ty = int(hit_point[0]), int(hit_point[1])
            cv2.circle(after_img, (tx, ty), 8, (0,0,255), 2)

        # Draw sector lines
        SECTOR_ANGLE = 18
        for i in range(20):
            ang = math.radians(i * SECTOR_ANGLE)
            x2 = int(center[0] + 1000 * math.cos(ang))
            y2 = int(center[1] - 1000 * math.sin(ang))
            cv2.line(after_img, center, (x2, y2), (255,0,0), 1)

        timestamp = datetime.datetime.now().strftime("%H%M%S")
        out_name = f"overlay_debug_{timestamp}.jpg"
        cv2.imwrite(out_name, after_img)
        print(f"Overlay+hit written to {out_name}")
        sys.exit(0)

    # ---------------------------
    # MODE: aruco
    # ---------------------------
    if len(sys.argv) >= 2 and sys.argv[1] == "aruco":
        if len(sys.argv) != 4:
            print("Usage: python3 detect_dart.py aruco BEFORE.jpg OUTPUT.jpg")
            sys.exit(1)

        before_path = sys.argv[2]
        out_path = sys.argv[3]
        before = load_image(before_path)

        # Show ArUco markers
        try:
            aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
            params = aruco.DetectorParameters_create()
            gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=params)

            if ids is not None:
                aruco.drawDetectedMarkers(before, corners, ids)
                print(f"[ARUCO] Found markers: {ids.flatten()}")
            else:
                print("[ARUCO] No markers detected.")
        except Exception as e:
            print(f"[ARUCO] error: {e}")

        cv2.imwrite(out_path, before)
        print(f"Saved ArUco debug to {out_path}")
        sys.exit(0)

    # ---------------------------
    # Default: detect impact
    # ---------------------------
    if len(sys.argv) != 3:
        print("Usage:")
        print("  python3 detect_dart.py BEFORE.jpg AFTER.jpg")
        print("  python3 detect_dart.py overlay INPUT.jpg OUTPUT.jpg")
        print("  python3 detect_dart.py overlayhit BEFORE.jpg AFTER.jpg OUTPUT.jpg")
        print("  python3 detect_dart.py aruco BEFORE.jpg OUTPUT.jpg")
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


def detect_impact(before_img, after_img):
    """
    High-level helper used by the app when the camera is live.
    Option A: images already rotated + warped in load_image().
    """
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
    """
    Same logic as find_dart_center, consistent with Option A warp pipeline.
    """
    tip, _ = find_dart_center(before_img, after_img, debug_img)
    if tip is None:
        print("[ERROR] No dart pixels found.")
        return None

    print(f"[DEBUG] Tip estimate: ({tip[0]:.1f}, {tip[1]:.1f})")
    return tip


if __name__ == "__main__":
    main()


# ------------------------------------------
# Utility for external callers (legacy use)
# ------------------------------------------
def get_board_sector_and_ring(x, y, board_center=(960,540)):
    dx = x - board_center[0]
    dy = y - board_center[1]
    r = math.hypot(dx, dy)
    r_frac = r / max(1.0, BOARD_RADIUS)

    angle = (math.degrees(math.atan2(dy, dx)) + ANGLE_OFFSET_DEGREES + 360) % 360
    SECTOR_ANGLE = 18

    ring = ring_from_radius_frac(r_frac)
    if ring == "miss":
        return None, ring

    sector_index = int((angle - 5) % 360 // SECTOR_ANGLE)
    return SECTORS[sector_index], ring
