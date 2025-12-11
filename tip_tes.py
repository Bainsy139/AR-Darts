import cv2
import numpy as np

# Approximate board centre in image coordinates (tweak if needed)
BOARD_CX = 1042
BOARD_CY = 625

before = cv2.imread("before.jpg")
after = cv2.imread("after.jpg")

if before is None or after is None:
    raise RuntimeError("before/after images missing")

# --- 1. Compute difference mask ---
# Boost the difference a bit so faint pixels (like the shaft) stand out more.
diff = cv2.absdiff(before, after)
gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

# Slight blur to connect nearby pixels along the shaft.
gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Lower threshold so we keep more of the subtle changes.
_, thresh = cv2.threshold(gray_blur, 10, 255, cv2.THRESH_BINARY)

# --- 2. Morphological cleanup ---
# First close small gaps (to join broken shaft pixels), then open to remove isolated noise.
kernel_close = np.ones((5, 5), np.uint8)
kernel_open = np.ones((3, 3), np.uint8)
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
mask = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

# Debug: save the binary mask so we can see if the shaft is included.
cv2.imwrite("tip_mask_debug.jpg", mask)

# --- 3. Find connected components ---
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

if num_labels <= 1:
    print("No dart blob detected.")
    exit()

# Largest component (ignore label 0 = background)
largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

ys, xs = np.where(labels == largest_label)
coords = np.column_stack((xs, ys))

# --- 4. Compute centroid of blob ---
# (centre of all pixels belonging to the largest connected component)
cx = float(np.mean(xs))
cy = float(np.mean(ys))
centroid = np.array([cx, cy], dtype=np.float32)

# --- 5. Use PCA to estimate dart orientation ---
# We treat the blob pixels as a point cloud and find the principal axis.
coords = np.column_stack((xs, ys)).astype(np.float32)
coords_centered = coords - centroid

# Guard against degenerate cases (very small or round blob)
if coords_centered.shape[0] >= 2:
    # SVD-based PCA: first right-singular vector gives principal direction in (x, y)
    _, _, vh = np.linalg.svd(coords_centered, full_matrices=False)
    direction = vh[0]  # shape (2,)

    # Normalise direction
    norm = np.linalg.norm(direction)
    if norm > 1e-6:
        direction = direction / norm
    else:
        direction = np.array([0.0, -1.0], dtype=np.float32)  # fallback
else:
    direction = np.array([0.0, -1.0], dtype=np.float32)

# Decide which way along the principal axis is "towards the tip".
# We assume the dart points roughly towards the board centre.
board_center = np.array([BOARD_CX, BOARD_CY], dtype=np.float32)
vec_to_center = board_center - centroid

if np.dot(direction, vec_to_center) < 0:
    direction = -direction

# Choose a projection length. This does not need to be exact for now; we mostly
# care about the angular direction. 80 px is a reasonable starting guess.
proj_len = 80.0

# Estimated tip by projecting from the centroid along the principal axis.
tip = centroid + direction * proj_len

tip_x, tip_y = int(round(tip[0])), int(round(tip[1]))

print("Centroid:", (cx, cy))
print("Direction vector (towards centre):", direction.tolist())
print("TIP ESTIMATE (PCA projection):", (tip_x, tip_y))

# --- 6. Draw results for visual testing ---
vis = after.copy()

# Draw all pixels in the detected blob so we can see the whole dart shape.
for (x, y) in coords:
    vis[int(y), int(x)] = (0, 255, 255)  # yellow blob pixels

# Mark centroid (green) and tip (red)
cv2.circle(vis, (int(round(cx)), int(round(cy))), 8, (0, 255, 0), -1)
cv2.circle(vis, (tip_x, tip_y), 8, (0, 0, 255), -1)

# Draw a line from centroid (green) to tip (red) for visual reference
cv2.line(vis, (int(round(cx)), int(round(cy))), (tip_x, tip_y), (255, 0, 0), 2)

cv2.imwrite("tip_debug.jpg", vis)
print("Saved tip_debug.jpg")
