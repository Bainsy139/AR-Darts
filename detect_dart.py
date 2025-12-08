import sys
import math
import cv2
import numpy as np

# ------------------------------
# CONFIG – tweak these as needed
# ------------------------------

# Board centre (pixels) and radius in the captured image
# TODO: put your real numbers here once you've measured from clean_board.jpg
BOARD_CX = 1042   # example for a 1920x1080 frame – change later
BOARD_CY = 620
BOARD_RADIUS = 170  # pixels from centre to outer board edge

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

# Threshold tuning
DIFF_BLUR_KSIZE = 9
DIFF_THRESHOLD = 25
MIN_BLOB_AREA = 30


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
    if CAMERA_UPSIDE_DOWN:
        img = cv2.rotate(img, cv2.ROTATE_180)
    return img


def preprocess_for_diff(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (DIFF_BLUR_KSIZE, DIFF_BLUR_KSIZE), 0)
    return gray


def find_dart_center(before_img, after_img):
    g_before = preprocess_for_diff(before_img)
    g_after = preprocess_for_diff(after_img)

    # Highlight areas that became darker in the AFTER frame
    # (dart hole / shadow should be darker than bare board)
    delta = cv2.subtract(g_before, g_after)

    _, mask = cv2.threshold(delta, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_center = None
    best_score = -1.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_BLOB_AREA:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        dx = cx - BOARD_CX
        dy = cy - BOARD_CY
        r = math.hypot(dx, dy)
        if r > BOARD_RADIUS * 1.2:
            continue

        r_frac = r / max(1.0, BOARD_RADIUS)
        score = area * (0.5 + r_frac)

        if score > best_score:
            best_score = score
            best_center = (cx, cy)

    return best_center, mask


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


def main():
    # Debug overlay mode:
    #   python3 detect_dart.py overlay INPUT.jpg OUTPUT.jpg
    if len(sys.argv) >= 2 and sys.argv[1] == "overlay":
        if len(sys.argv) != 4:
            print("Usage: python3 detect_dart.py overlay INPUT.jpg OUTPUT.jpg")
            sys.exit(1)
        input_path = sys.argv[2]
        output_path = sys.argv[3]
        draw_debug_overlay(input_path, output_path)
        print(f"Overlay written to {output_path}")
        sys.exit(0)

    # Default CLI mode:
    #   python3 detect_dart.py BEFORE.jpg AFTER.jpg
    if len(sys.argv) != 3:
        print("Usage:")
        print("  python3 detect_dart.py BEFORE.jpg AFTER.jpg")
        print("  python3 detect_dart.py overlay INPUT.jpg OUTPUT.jpg")
        sys.exit(1)

    before_path = sys.argv[1]
    after_path = sys.argv[2]

    before = load_image(before_path)
    after = load_image(after_path)

    center, mask = find_dart_center(before, after)

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
