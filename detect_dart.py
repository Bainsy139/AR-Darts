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
ROT_OFFSET_DEG = -9.8

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
USE_ARUCO_WARP = True  # try to refine the warp matrix from ArUco markers if available

# Default source points: manual estimate of the outer double ring top/right/bottom/left.
# These are used as a fallback when ArUco-based calibration is not available.
DEFAULT_SRC_POINTS = np.float32([
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

# This will be filled on first use, either from ArUco markers or from DEFAULT_SRC_POINTS.
WARP_MATRIX = None

# Threshold tuning
DIFF_BLUR_KSIZE = 9
DIFF_THRESHOLD = 15
MIN_BLOB_AREA = 10

# Tip-biased edge-diff tuning (no ML, single before/after)
HP_BLUR_KSIZE = 41          # large blur to remove low-frequency illumination changes
CANNY_LOW = 40              # edge thresholds for high-pass diff
CANNY_HIGH = 120
EDGE_DILATE_ITERS = 1       # slightly thicken edges so we get a stable point
MIN_EDGE_PIXELS = 8        # minimum edge pixels to accept a detection
TIP_K_CLOSEST = 25          # average of K most-inward edge pixels

# Ray-constrained tip selection (for picking the actual dart tip, not shaft/flight edge)
RAY_BAND_PX = 10            # max perpendicular distance (px) from the centre→centroid ray to consider as “dart-aligned”
MIN_RAY_PIXELS = 15         # min pixels on that ray band to trust the ray-based tip

# Tip selection tuning
TIP_NUDGE_PX = 2            # after selecting the inward endpoint, nudge slightly further toward board centre
COMP_DILATE_ITERS = 2       # dilate the coarse diff blob so edge pixels from the dart are included

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
        global WARP_MATRIX
        h, w = img.shape[:2]

        # Lazily initialise the warp matrix on first use.
        if WARP_MATRIX is None:
            M = _compute_warp_from_aruco(img)
            if M is None:
                # ArUco not available or markers not found → fall back to manual points.
                M = cv2.getPerspectiveTransform(DEFAULT_SRC_POINTS, DST_POINTS)
            WARP_MATRIX = M

        img = cv2.warpPerspective(img, WARP_MATRIX, (w, h))

    return img


def preprocess_for_diff(img):
    # Keep this light; we do the heavy blurs on the diff image so we don’t erase the tip.
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Try to compute a perspective warp from 4 ArUco markers.
def _compute_warp_from_aruco(img):
    """Try to compute a perspective warp from 4 ArUco markers.

    Expects marker IDs 0,1,2,3 placed at (roughly) top, right, bottom, left
    of the outer double ring. Returns a 3x3 warp matrix or None.
    """
    if not HAS_ARUCO or not USE_ARUCO_WARP:
        return None

    # Older OpenCV ArUco API style
    try:
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        parameters = aruco.DetectorParameters_create()
    except AttributeError:
        # If the ArUco module is not fully available, bail out cleanly.
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=parameters)

    if ids is None or len(ids) < 4:
        # Not all markers visible
        return None

    ids = ids.flatten()
    required_ids = [0, 1, 2, 3]
    if not all(r in ids for r in required_ids):
        return None

    # Build src points in a fixed logical order: top, right, bottom, left.
    src_pts = []
    for marker_id in required_ids:
        idx = int(np.where(ids == marker_id)[0][0])
        # Use the centre of the marker for robustness
        c = corners[idx][0].mean(axis=0)
        src_pts.append(c)

    src_pts = np.array(src_pts, dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_pts, DST_POINTS)
    return M


def _tip_from_pca_endpoints(coords: np.ndarray, board_center: np.ndarray):
    """Return the inward endpoint along the major axis of a set of (x,y) points.

    coords: float32 array shape (N,2) in (x,y)
    board_center: float32 array shape (2,)

    Returns: (tip_xy, axis_unit) where tip_xy is float32 (2,) and axis_unit is float32 (2,)
    """
    if coords is None or len(coords) < 5:
        return None, None

    pts = coords.astype(np.float32)
    mean = pts.mean(axis=0)
    centered = pts - mean

    # PCA via covariance eigenvectors
    cov = (centered.T @ centered) / max(1.0, float(len(pts)))
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))].astype(np.float32)

    axis_norm = float(np.hypot(axis[0], axis[1]))
    if axis_norm < 1e-6:
        return None, None

    u = axis / axis_norm  # unit major axis

    proj = centered @ u
    pmin = float(np.min(proj))
    pmax = float(np.max(proj))

    end1 = mean + u * pmin
    end2 = mean + u * pmax

    # Vector from centroid toward board centre
    vc = board_center - mean
    vc_norm = float(np.hypot(vc[0], vc[1]))
    if vc_norm < 1e-6:
        return None, None
    vc /= vc_norm

    v1 = end1 - mean
    v2 = end2 - mean

    dot1 = float(np.dot(v1, vc))
    dot2 = float(np.dot(v2, vc))

    # Only accept endpoints that actually point toward the board
    candidates = []
    if dot1 > 0:
        candidates.append((dot1, end1))
    if dot2 > 0:
        candidates.append((dot2, end2))

    if not candidates:
        # PCA axis unreliable → force fallback
        return None, None

    # Choose the endpoint most aligned with board centre
    _, tip = max(candidates, key=lambda x: x[0])

    return tip.astype(np.float32), u.astype(np.float32)


def find_dart_center(before_img, after_img, debug_img=None):
    g_before = preprocess_for_diff(before_img)
    g_after = preprocess_for_diff(after_img)

    # 1) Absolute difference (captures both darker and brighter changes)
    diff = cv2.absdiff(g_before, g_after)

    # 2) Restrict to plausible board area early (prevents off-board noise from winning).
    h, w = diff.shape
    yy, xx = np.mgrid[0:h, 0:w]
    dx = xx - BOARD_CX
    dy = yy - BOARD_CY
    r = np.sqrt(dx * dx + dy * dy)
    board_mask = (r <= BOARD_RADIUS * 1.1)
    diff = np.where(board_mask, diff, 0).astype(np.uint8)

    # 3) Coarse blob from diff to localise the dart change.
    #    This is intentionally "fat" and tolerant; we only use it to gate edge pixels later.
    diff_blur = cv2.GaussianBlur(diff, (DIFF_BLUR_KSIZE, DIFF_BLUR_KSIZE), 0)
    _, diff_bin = cv2.threshold(diff_blur, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    # Close then open (fill gaps, remove speckle)
    k = np.ones((5, 5), np.uint8)
    diff_bin = cv2.morphologyEx(diff_bin, cv2.MORPH_CLOSE, k, iterations=1)
    diff_bin = cv2.morphologyEx(diff_bin, cv2.MORPH_OPEN, k, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(diff_bin, connectivity=8)

    # Pick the largest component inside the board (ignore background label 0)
    best_label = None
    best_area = 0
    for lab in range(1, num_labels):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < MIN_BLOB_AREA:
            continue
        if area > best_area:
            best_area = area
            best_label = lab

    if best_label is None:
        # Still return an image for debug callers
        if debug_img is not None:
            cv2.imwrite("debug_last_blob.jpg", debug_img)
        return None, diff_bin

    comp = (labels == best_label).astype(np.uint8) * 255

    # Draw largest blob contour if requested
    if debug_img is not None:
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            comp_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(debug_img, [comp_contour], -1, (0, 255, 0), 2)

    if COMP_DILATE_ITERS > 0:
        comp = cv2.dilate(comp, np.ones((3, 3), np.uint8), iterations=COMP_DILATE_ITERS)

    # 4) High-pass the diff to reduce projector/illumination drift.
    hp = diff.astype(np.float32) - cv2.GaussianBlur(diff, (HP_BLUR_KSIZE, HP_BLUR_KSIZE), 0).astype(np.float32)
    hp = np.clip(hp, 0, 255).astype(np.uint8)

    # Gate hp to the coarse component mask so UI/lighting edges don't dominate
    hp = cv2.bitwise_and(hp, hp, mask=comp)

    # 5) Edges of the gated high-pass diff.
    edges = cv2.Canny(hp, CANNY_LOW, CANNY_HIGH)

    if EDGE_DILATE_ITERS > 0:
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=EDGE_DILATE_ITERS)

    ys, xs = np.where(edges > 0)
    print(f"[DEBUG] Found {len(xs)} edge pixels after Canny")
    if len(xs) < MIN_EDGE_PIXELS:
        if debug_img is not None:
            cv2.imwrite("debug_last_blob.jpg", debug_img)
        return None, edges

    coords = np.column_stack((xs, ys)).astype(np.float32)
    c = np.array([BOARD_CX, BOARD_CY], dtype=np.float32)

    # 6) Primary tip heuristic: major-axis inward endpoint (PCA)
    tip = None
    axis_u = None
    tip_pca, axis_u = _tip_from_pca_endpoints(coords, c)
    if tip_pca is not None:
        # Nudge a little further toward the board centre (helps when the endpoint is the shaft edge)
        v = c - tip_pca
        vn = float(np.hypot(v[0], v[1]))
        if vn > 1e-6:
            tip = tip_pca + (v / vn) * float(TIP_NUDGE_PX)
        else:
            tip = tip_pca

    # 7) Secondary: ray-constrained inward point (keeps behaviour when PCA is unstable)
    if tip is None:
        centroid = coords.mean(axis=0)
        v = centroid - c
        v_norm = float(np.hypot(v[0], v[1]))
        if v_norm > 1e-6:
            d = v / v_norm  # unit direction from centre outward
            rel = coords - c
            t = rel[:, 0] * d[0] + rel[:, 1] * d[1]
            perp = np.abs(rel[:, 0] * (-d[1]) + rel[:, 1] * d[0])
            mask = (t > 0) & (perp <= RAY_BAND_PX)
            cand = coords[mask]
            cand_t = t[mask]
            if len(cand) >= MIN_RAY_PIXELS:
                tip = cand[int(np.argmin(cand_t))]

    # 8) Final fallback: average of K closest edge pixels to centre
    if tip is None:
        d2 = np.sum((coords - c) ** 2, axis=1)
        k = min(TIP_K_CLOSEST, len(d2))
        idxs = np.argpartition(d2, k - 1)[:k]
        tip = coords[idxs].mean(axis=0)

    # Draw tip if requested
    if debug_img is not None and tip is not None:
        cv2.circle(debug_img, (int(round(tip[0])), int(round(tip[1]))), 5, (0, 0, 255), -1)

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

        # Load images without warp for ArUco detection
        before = cv2.imread(before_path, cv2.IMREAD_COLOR)
        after = cv2.imread(after_path, cv2.IMREAD_COLOR)
        overlay = cv2.imread(after_path, cv2.IMREAD_COLOR)

        # Apply camera orientation correction if needed
        if CAMERA_UPSIDE_DOWN:
            before = cv2.rotate(before, cv2.ROTATE_180)
            after = cv2.rotate(after, cv2.ROTATE_180)
            overlay = cv2.rotate(overlay, cv2.ROTATE_180)

        # Compute warp matrix (possibly from ArUco)
        h, w = before.shape[:2]
        M = _compute_warp_from_aruco(before)
        if M is None:
            M = cv2.getPerspectiveTransform(DEFAULT_SRC_POINTS, DST_POINTS)
        if M is not None:
            warped_before = cv2.warpPerspective(before, M, (w, h))
            warped_after = cv2.warpPerspective(after, M, (w, h))
            warped_overlay = cv2.warpPerspective(overlay, M, (w, h))
            # Use estimate_tip with warped images
            hit_point = estimate_tip(warped_before, warped_after, warped_overlay)
            print(f"DEBUG: Estimated tip @ {hit_point}")
            draw_debug_overlay_with_hit(after_path, hit_point, out_path)
            print(f"Overlay+hit written to {out_path}")
            sys.exit(0)
        else:
            print("Could not compute perspective warp, aborting overlayhit.")
            sys.exit(1)

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



def estimate_tip(before_img, after_img, debug_img=None):
    """
    Estimate the dart tip location using the same logic as find_dart_center,
    but returns only the tip coordinates (x, y) or None.
    Optionally draws debug info on debug_img if provided.
    """
    result, _ = find_dart_center(before_img, after_img, debug_img)
    if result is None:
        return None
    return result

if __name__ == "__main__":
    main()