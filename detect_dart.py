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
BOARD_CY = 692
BOARD_RADIUS = 520  # pixels from centre to outer board edge

# Rotation offset in degrees to align sector 20 to the top
ROT_OFFSET_DEG = -18.0

# Rough ring ratios – we’ll refine later
def ring_from_radius_frac(r_frac: float) -> str:
    if r_frac <= 0.05:
        return "inner_bull"
    if r_frac <= 0.12:
        return "outer_bull"
    if 0.85 <= r_frac <= 0.95:
        return "double"
    if 0.5 <= r_frac <= 0.6:
        return "treble"
    if r_frac > 1.02:
        return "miss"
    return "single"

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


def classify_hit(x: float, y: float):
    r_frac, angle = pixel_to_polar(x, y)

    # DEBUG: print raw polar values so we can calibrate rings/rotation later
    angle_deg = math.degrees(angle)
    print(f"[DEBUG] r_frac={r_frac:.3f}, angle_deg={angle_deg:.1f}")

    ring = ring_from_radius_frac(r_frac)
    if ring == "miss":
        return ring, None

    sec_idx = sector_index_from_angle(angle)
    sector_num = SECTORS[sec_idx]
    return ring, sector_num


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


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 detect_dart.py BEFORE.jpg AFTER.jpg")
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
