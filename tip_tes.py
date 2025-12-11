import cv2
import numpy as np

# --- Board geometry (must match detect_dart.py) ---
BOARD_CX = 1042
BOARD_CY = 625
BOARD_RADIUS = 180  # outer double radius in pixels – adjust if needed

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

# --- 4. Connected components to find the dart blob ---
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

if num_labels <= 1:
    print("No dart blob detected.")
    cv2.imwrite("tip_debug.jpg", after)
    raise SystemExit

largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
ys, xs = np.where(labels == largest_label)
coords = np.column_stack((xs, ys)).astype(np.float32)

# --- 5. Centroid of the blob ---
cx = float(np.mean(xs))
cy = float(np.mean(ys))
centroid = np.array([cx, cy], dtype=np.float32)

# --- 6. PCA for orientation ---
coords_centered = coords - centroid

if coords_centered.shape[0] >= 2:
    _, _, vh = np.linalg.svd(coords_centered, full_matrices=False)
    direction = vh[0]
    norm = np.linalg.norm(direction)
    if norm > 1e-6:
        direction = direction / norm
    else:
        direction = np.array([0.0, -1.0], dtype=np.float32)
else:
    direction = np.array([0.0, -1.0], dtype=np.float32)

# Orient the direction so it points roughly towards the board centre
board_center = np.array([BOARD_CX, BOARD_CY], dtype=np.float32)
vec_to_center = board_center - centroid
if np.dot(direction, vec_to_center) < 0:
    direction = -direction

# --- 7. Ray-march from flight towards board centre on raw diff ---
max_len = 220
thresh_val = 15

gray_raw = gray  # already grayscale diff

candidate_tip = None

for t in range(int(max_len)):
    px = int(round(cx + direction[0] * t))
    py = int(round(cy + direction[1] * t))

    if px < 0 or px >= w or py < 0 or py >= h:
        break

    # Stay within the board mask only
    if mask_board[py, px] == 0:
        break

    val = gray_raw[py, px]
    if val >= thresh_val:
        candidate_tip = (px, py)

if candidate_tip is not None:
    tip_x, tip_y = candidate_tip
else:
    proj_len = 80.0
    tip = centroid + direction * proj_len
    tip_x, tip_y = int(round(tip[0])), int(round(tip[1]))

print("Centroid:", (cx, cy))
print("Direction (towards centre):", direction.tolist())
print("Estimated TIP:", (tip_x, tip_y))

# --- 8. Draw visual debug ---
vis = after.copy()

# Draw all blob pixels in yellow so we see the dart mass
for (x, y) in coords:
    vis[int(y), int(x)] = (0, 255, 255)

# Centroid (green) and tip (red)
cv2.circle(vis, (int(round(cx)), int(round(cy))), 8, (0, 255, 0), -1)
cv2.circle(vis, (tip_x, tip_y), 8, (0, 0, 255), -1)

# Line from centroid to tip (blue)
cv2.line(vis, (int(round(cx)), int(round(cy))), (tip_x, tip_y), (255, 0, 0), 2)

cv2.imwrite("tip_debug.jpg", vis)
print("Saved tip_debug.jpg")
