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
diff = cv2.absdiff(before, after)
gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)

# --- 2. Morphological cleanup ---
kernel = np.ones((5,5), np.uint8)
mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

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
cx = np.mean(xs)
cy = np.mean(ys)

# --- 5. Estimate tip as the point in the blob closest to the board centre ---
# We assume the dart tip is nearer the board centre than the flight.
# Compute squared radial distance from the known board centre for each pixel.
r2 = (xs - BOARD_CX) ** 2 + (ys - BOARD_CY) ** 2
idx = np.argmin(r2)
tip_x, tip_y = coords[idx]

print("Centroid:", (float(cx), float(cy)))
print("TIP ESTIMATE (closest to board centre):", (int(tip_x), int(tip_y)))

# --- 6. Draw results for visual testing ---
vis = after.copy()
cv2.circle(vis, (int(cx), int(cy)), 8, (0,255,0), -1)    # green = centroid
cv2.circle(vis, (int(tip_x), int(tip_y)), 8, (0,0,255), -1)  # red = estimated tip

# Draw a line from centroid (green) to tip (red) for visual reference
cv2.line(vis, (int(cx), int(cy)), (int(tip_x), int(tip_y)), (255, 0, 0), 2)

cv2.imwrite("tip_debug.jpg", vis)
print("Saved tip_debug.jpg")