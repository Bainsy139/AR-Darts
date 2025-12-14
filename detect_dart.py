import sys
import math
import cv2
import numpy as np

# ------------------------------
# CONFIG – tweak these as needed
# ------------------------------

# Board centre (pixels) and radius in the captured image
# Calibrated from overlay_debug on 1920x1080 frames
BOARD_CX = 1042   # horizontal centre of board
BOARD_CY = 625    # vertical centre of board
BOARD_RADIUS = 130  # pixels from centre to outer double ring edge (tweak if overlay drifts)

# Rotation offset in degrees to align sector 20 to the top
# Previously -18.0, but tests show we were off by one full wedge (18°).
# Using 0.0 brings 20/1/5/19/15 etc into the correct sectors.
ROT_OFFSET_DEG = -7.8

# Rough ring ratios – we’ll refine later
def ring_from_radius_frac(r_frac: float) -> str:
    # Inner bull (0 – 0.035)
    if r_frac <= 0.035:
        return "inner_bull"

    # Outer bull (0.035 – 0.09)
    if r_frac <= 0.09:
        return "outer_bull"

    # Single (inner segment) up to start of treble
    if r_frac < 0.57:
        return "single"

    # Treble (0.57 – 0.63)
    if r_frac <= 0.63:
        return "treble"

    # Single (outer segment) up to inner double
    if r_frac < 0.95:
        return "single"

    # Double (0.95 – 1.0)
    if r_frac <= 1.0:
        return "double"

    # Beyond board = miss
    return "miss"

# Sector order (clockwise from 20 at 12 o'clock)
SECTORS = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17,
           3, 19, 7, 16, 8, 11, 14, 9, 12, 5]

# If camera is upside-down
CAMERA_UPSIDE_DOWN = True

# Optional perspective warp to correct board foreshortening.
# We now calibrate this from 4 points on the outer double ring.
# Source points (from overlay_local_test.jpg, AFTER rotation):
#   top, right, bottom, left
USE_WARP = True

SRC_POINTS = np.float32([
    [1039, 483],   # top
    [1195, 617],   # right
    [1045, 738],   # bottom
    [890,  627],   # left
])

# Target points: where those 4 points would be on a perfect circle
# centred at (BOARD_CX, BOARD_CY) with radius BOARD_RADIUS.
DST_POINTS = np.float32([
    [BOARD_CX,               BOARD_CY - BOARD_RADIUS],  # top
    [BOARD_CX + BOARD_RADIUS, BOARD_CY],                # right
    [BOARD_CX,               BOARD_CY + BOARD_RADIUS],  # bottom
    [BOARD_CX - BOARD_RADIUS, BOARD_CY],                # left
])

WARP_MATRIX = cv2.getPerspectiveTransform(SRC_POINTS, DST_POINTS)

# Threshold tuning
DIFF_BLUR_KSIZE = 9
DIFF_THRESHOLD = 25
MIN_BLOB_AREA = 30

# Tip-biased edge-diff tuning (no ML, single before/after)
HP_BLUR_KSIZE = 41          # large blur to remove low-frequency illumination changes
CANNY_LOW = 40              # edge thresholds for high-pass diff
CANNY_HIGH = 120
EDGE_DILATE_ITERS = 1       # slightly thicken edges so we get a stable point
MIN_EDGE_PIXELS = 25        # minimum edge pixels to accept a detection
TIP_K_CLOSEST = 25          # average of K most-inward edge pixels


def sector_index_from_angle(angle: float) -> int:
    """Match the JS sector indexing logic."""
    rot_rad = math.radians(ROT_OFFSET_DEG)
    a = angle + math.pi / 2 - rot_rad
    two_pi = 2 * math.pi
    a = (a % two_pi + two_pi) % two_pi
    idx = int(math.floor(a / two_pi * 20)) % 20
    return idx


def pixel_to_polar(x: float, y: float):
    dx = x - BOARD_CX
    dy = y - BOARD_CY
    r = math.hypot(dx, dy)
    r_frac = r / max(1.0, BOARD_RADIUS)
    angle = math.atan2(dy, dx)
    return r_frac, angle


def classify_hit_with_debug(x: float, y: float):
    """Classify a hit and return extra debug info for calibration.

    Returns a dict with keys: ring, sector, r_frac, angle_deg.
    """
    r_frac, angle = pixel_to_polar(x, y)
    angle_deg = math.degrees(angle)

    ring = ring_from_radius_frac(r_frac)
    if ring == "miss":
        sector = None
    else:
        sec_idx = sector_index_from_angle(angle)
        sector = SECTORS[sec_idx]

    return {
        "ring": ring,
        "sector": sector,
        "r_frac": r_frac,
        "angle_deg": angle_deg,
    }


def classify_hit(x: float, y: float):
    """Backwards-compatible wrapper used by the CLI script.

    Prints debug info and returns (ring, sector) as before.
    """
    info = classify_hit_with_debug(x, y)
    print(f"[DEBUG] r_frac={info['r_frac']:.3f}, angle_deg={info['angle_deg']:.1f}")
    return info["ring"], info["sector"]


def load_image(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to load image: {path}")

    # First apply camera orientation correction.
    if CAMERA_UPSIDE_DOWN:
        img = cv2.rotate(img, cv2.ROTATE_180)

    # Then optionally apply perspective warp to correct foreshortening.
    if USE_WARP:
        h, w = img.shape[:2]
        img = cv2.warpPerspective(img, WARP_MATRIX, (w, h))

    return img


def preprocess_for_diff(img):
    # Keep this light; we do the heavy blurs on the diff image so we don’t erase the tip.
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def find_dart_center(before_img, after_img):
    g_before = preprocess_for_diff(before_img)
    g_after = preprocess_for_diff(after_img)

    # 1) Absolute difference (captures both darker and brighter changes)
    diff = cv2.absdiff(g_before, g_after)

    # 2) High-pass the diff to reduce projector/illumination drift.
    #    Remove low-frequency changes (like soft shadows / exposure) while keeping sharp structure (dart edges).
    hp = diff.astype(np.float32) - cv2.GaussianBlur(diff, (HP_BLUR_KSIZE, HP_BLUR_KSIZE), 0).astype(np.float32)
    hp = np.clip(hp, 0, 255).astype(np.uint8)

    # 3) Restrict to plausible board area early (prevents off-board noise from winning).
    h, w = hp.shape
    yy, xx = np.mgrid[0:h, 0:w]
    dx = xx - BOARD_CX
    dy = yy - BOARD_CY
    r = np.sqrt(dx * dx + dy * dy)
    board_mask = (r <= BOARD_RADIUS * 1.1)
    hp = np.where(board_mask, hp, 0).astype(np.uint8)

    # 4) Edges of the high-pass diff. This suppresses “fat” blobs from flights/shadows
    #    and prefers sharp boundaries (shaft/tip/wire).
    edges = cv2.Canny(hp, CANNY_LOW, CANNY_HIGH)

    if EDGE_DILATE_ITERS > 0:
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=EDGE_DILATE_ITERS)

    ys, xs = np.where(edges > 0)
    if len(xs) < MIN_EDGE_PIXELS:
        return None, edges

    coords = np.column_stack((xs, ys)).astype(np.float32)

    # 5) Tip heuristic: choose the most inward edge pixels (closest to board centre)
    c = np.array([BOARD_CX, BOARD_CY], dtype=np.float32)
    d2 = np.sum((coords - c) ** 2, axis=1)

    k = min(TIP_K_CLOSEST, len(d2))
    idxs = np.argpartition(d2, k - 1)[:k]
    tip = coords[idxs].mean(axis=0)

    return (float(tip[0]), float(tip[1])), edges


def detect_impact(before_img, after_img):
    """High-level helper: given BEFORE and AFTER BGR images, return hit info.

    Returns a dict with keys:
      - hit: dict with ring, sector, r_frac, angle_deg, x, y
      - reason: string if no impact ("no_impact"), else None
    """
    center, _ = find_dart_center(before_img, after_img)

    if center is None:
        return {"hit": None, "reason": "no_impact"}

    cx, cy = center
    info = classify_hit_with_debug(cx, cy)
    info["x"] = cx
    info["y"] = cy

    # If it landed off the board, treat as miss
    if info["ring"] == "miss" or info["sector"] is None:
        return {"hit": None, "reason": "off_board"}

    return {"hit": info, "reason": None}


def draw_debug_overlay(input_path: str, output_path: str):
    """
    Debug helper: draw our current idea of the board geometry (rings + wedge lines)
    onto a camera frame so we can visually tune BOARD_CX/BOARD_CY/BOARD_RADIUS/ROT_OFFSET_DEG.

    Usage from CLI (see main):
        python3 detect_dart.py overlay INPUT.jpg OUTPUT.jpg
    """
    img = load_image(input_path)
    # Work on a copy so we don't mutate the original
    overlay = img.copy()

    # Centre as integer pixel coords
    center = (int(round(BOARD_CX)), int(round(BOARD_CY)))

    # Draw outer board edge
    cv2.circle(overlay, center, int(round(BOARD_RADIUS)), (0, 0, 255), 2)

    # Draw the main ring boundaries based on the same fractions used in ring_from_radius_frac
    ring_fracs = [0.035, 0.09, 0.57, 0.63, 0.95]
    for frac in ring_fracs:
        r = int(round(BOARD_RADIUS * frac))
        cv2.circle(overlay, center, r, (0, 255, 0), 1)

    # Draw the 20 wedge boundaries using the same rotation convention as sector_index_from_angle
    rot_rad = math.radians(ROT_OFFSET_DEG)
    two_pi = 2.0 * math.pi
    for k in range(20):
        # Boundary angles are where the sector index changes; these are spaced every 18°
        angle = -math.pi / 2 + rot_rad + (k * two_pi / 20.0)
        x2 = int(round(BOARD_CX + BOARD_RADIUS * math.cos(angle)))
        y2 = int(round(BOARD_CY + BOARD_RADIUS * math.sin(angle)))
        cv2.line(overlay, center, (x2, y2), (255, 0, 0), 1)

    # Mark the centre point explicitly
    cv2.circle(overlay, center, 3, (255, 255, 255), -1)

    cv2.imwrite(output_path, overlay)

def draw_debug_overlay_with_hit(input_path: str, hit_xy, output_path: str):
    img = load_image(input_path)
    overlay = img.copy()

    center = (int(round(BOARD_CX)), int(round(BOARD_CY)))

    cv2.circle(overlay, center, int(round(BOARD_RADIUS)), (0, 0, 255), 2)

    ring_fracs = [0.035, 0.09, 0.57, 0.63, 0.95]
    for frac in ring_fracs:
        r = int(round(BOARD_RADIUS * frac))
        cv2.circle(overlay, center, r, (0, 255, 0), 1)

    rot_rad = math.radians(ROT_OFFSET_DEG)
    two_pi = 2.0 * math.pi
    for k in range(20):
        angle = -math.pi / 2 + rot_rad + (k * two_pi / 20.0)
        x2 = int(round(BOARD_CX + BOARD_RADIUS * math.cos(angle)))
        y2 = int(round(BOARD_CY + BOARD_RADIUS * math.sin(angle)))
        cv2.line(overlay, center, (x2, y2), (255, 0, 0), 1)

    cv2.circle(overlay, center, 3, (255, 255, 255), -1)

    if hit_xy is not None:
        hx, hy = hit_xy
        pt = (int(round(hx)), int(round(hy)))
        cv2.circle(overlay, pt, 12, (255, 255, 255), 2)  # white outline
        cv2.circle(overlay, pt, 10, (0, 0, 255), -1)     # red fill
        cv2.circle(overlay, pt, 2, (0, 0, 0), -1)        # black center

    cv2.imwrite(output_path, overlay)


def main():
    # Debug overlay mode:
    #   python3 detect_dart.py overlay INPUT.jpg OUTPUT.jpg
    # Overlay + hit marker mode:
    #   python3 detect_dart.py overlayhit BEFORE.jpg AFTER.jpg OUTPUT.jpg
    if len(sys.argv) >= 2 and sys.argv[1] == "overlay":
        if len(sys.argv) != 4:
            print("Usage: python3 detect_dart.py overlay INPUT.jpg OUTPUT.jpg")
            sys.exit(1)

        input_path = sys.argv[2]
        out_path = sys.argv[3]
        draw_debug_overlay(input_path, out_path)
        print(f"Overlay written to {out_path}")
        sys.exit(0)

    if len(sys.argv) >= 2 and sys.argv[1] == "overlayhit":
        if len(sys.argv) != 5:
            print("Usage: python3 detect_dart.py overlayhit BEFORE.jpg AFTER.jpg OUTPUT.jpg")
            sys.exit(1)

        before_path = sys.argv[2]
        after_path = sys.argv[3]
        out_path = sys.argv[4]

        before = load_image(before_path)
        after = load_image(after_path)

        center, _ = find_dart_center(before, after)
        draw_debug_overlay_with_hit(after_path, center, out_path)
        print(f"Overlay+hit written to {out_path}")
        sys.exit(0)

    # Default CLI mode:
    #   python3 detect_dart.py BEFORE.jpg AFTER.jpg
    if len(sys.argv) != 3:
        print("Usage:")
        print("  python3 detect_dart.py BEFORE.jpg AFTER.jpg")
        print("  python3 detect_dart.py overlay INPUT.jpg OUTPUT.jpg")
        print("  python3 detect_dart.py overlayhit BEFORE.jpg AFTER.jpg OUTPUT.jpg")
        sys.exit(1)

    before_path = sys.argv[1]
    after_path = sys.argv[2]

    before = load_image(before_path)
    after = load_image(after_path)

    center, _ = find_dart_center(before, after)

    if center is None:
        print("No clear dart impact detected.")
        sys.exit(0)

    cx, cy = center
    ring, sector = classify_hit(cx, cy)

    if ring == "miss" or sector is None:
        print(f"Detected impact at ({cx:.1f}, {cy:.1f}) but classified as MISS.")
    else:
        if "bull" in ring:
            label = "inner bull" if ring == "inner_bull" else "outer bull"
        else:
            label = f"{ring} {sector}"

        print(f"Detected impact at ({cx:.1f}, {cy:.1f}) → {label}")


if __name__ == "__main__":
    main()
