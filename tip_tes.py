import cv2
import numpy as np

# --- Board geometry (must match detect_dart.py) ---
BOARD_CX = 1042
BOARD_CY = 625
BOARD_RADIUS = 180  # outer double radius in pixels – adjust if needed
TIP_K_CLOSEST = 25  # average of K closest blob pixels to centre to stabilise tip
TIP_USE_ERODE = True
TIP_ERODE_ITERS = 1  # 0/1/2; higher = more robust to wispy flight edges but may shrink too much

before = cv2.imread("before.jpg")
after = cv2.imread("after.jpg")

if before is None or after is None:
    raise RuntimeError("before/after images missing")

h, w = before.shape[:2]

# --- 1. Compute raw difference and grayscale ---
diff = cv2.absdiff(before, after)
gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

# --- 2. Build a circular board mask so we ignore projector junk off the board ---
mask_board = np.zeros((h, w), dtype=np.uint8)
cv2.circle(mask_board, (BOARD_CX, BOARD_CY), BOARD_RADIUS, 255, thickness=-1)

# Optional: erode slightly so we stay just inside the outer double
mask_board = cv2.erode(mask_board, np.ones((5, 5), np.uint8), iterations=1)

# --- 3. Threshold + morphology, but only inside the board mask ---
gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(gray_blur, 10, 255, cv2.THRESH_BINARY)

# Apply board mask so nothing outside the dartboard is considered
thresh = cv2.bitwise_and(thresh, thresh, mask=mask_board)

kernel_close = np.ones((5, 5), np.uint8)
kernel_open = np.ones((3, 3), np.uint8)
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
mask = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

cv2.imwrite("tip_mask_debug.jpg", mask)
# For debugging: the selected component mask (after optional erosion)
# (Written later after component selection once comp_mask exists)

# --- 4. Connected components to find the dart blob ---
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

if num_labels <= 1:
    print("No dart blob detected.")
    cv2.imwrite("tip_debug.jpg", after)
    raise SystemExit

largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
ys, xs = np.where(labels == largest_label)
coords = np.column_stack((xs, ys)).astype(np.float32)

# Component-only mask (for optional erosion / debugging)
comp_mask = np.zeros((h, w), dtype=np.uint8)
comp_mask[ys, xs] = 255

if TIP_USE_ERODE and TIP_ERODE_ITERS > 0:
    k = np.ones((3, 3), np.uint8)
    comp_mask = cv2.erode(comp_mask, k, iterations=TIP_ERODE_ITERS)

# Refresh coords after erosion (if erosion removed everything, fall back to original coords)
ys_e, xs_e = np.where(comp_mask == 255)
if len(xs_e) > 0:
    xs_use, ys_use = xs_e, ys_e
    coords_use = np.column_stack((xs_use, ys_use)).astype(np.float32)
else:
    xs_use, ys_use = xs, ys
    coords_use = coords

cv2.imwrite("tip_component_debug.jpg", comp_mask)

# --- 5. Board centre ---
board_center = np.array([BOARD_CX, BOARD_CY], dtype=np.float32)

# --- 6. Tip estimate: closest-to-centre blob pixel (Option A) ---
# Your constraint: the flight is always further from board centre than the tip.
# So we pick the point on the detected blob that minimises distance to the board centre.

# Compute distances for all blob pixels we are using (possibly eroded)
dx = coords_use[:, 0] - board_center[0]
dy = coords_use[:, 1] - board_center[1]
d2 = dx * dx + dy * dy

# Take the K closest pixels and average them (reduces jitter / single-pixel noise)
K = int(max(1, min(TIP_K_CLOSEST, len(d2))))
idx = np.argpartition(d2, K - 1)[:K]

tip_pt = np.mean(coords_use[idx], axis=0)
tip_x, tip_y = int(round(float(tip_pt[0]))), int(round(float(tip_pt[1])))

# Still compute centroid for debugging/visualisation (on the *original* component pixels)
cx = float(np.mean(xs))
cy = float(np.mean(ys))
centroid = np.array([cx, cy], dtype=np.float32)

print("Centroid:", (cx, cy))
print("Estimated TIP (closest-to-centre):", (tip_x, tip_y))

# --- 8. Draw visual debug ---
vis = after.copy()

# Draw all blob pixels in yellow so we see the dart mass
for (x, y) in coords_use:
    vis[int(y), int(x)] = (0, 255, 255)

# Centroid (green) and tip (red)
cv2.circle(vis, (int(round(cx)), int(round(cy))), 8, (0, 255, 0), -1)
cv2.circle(vis, (tip_x, tip_y), 8, (0, 0, 255), -1)

# Line from centroid to tip (blue)
cv2.line(vis, (int(round(cx)), int(round(cy))), (tip_x, tip_y), (255, 0, 0), 2)

cv2.imwrite("tip_debug.jpg", vis)
print("Saved tip_debug.jpg")
