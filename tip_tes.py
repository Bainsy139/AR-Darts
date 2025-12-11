import cv2
import numpy as np

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

# --- 5. Find furthest pixel from centroid ---
dists = np.sqrt((xs - cx)**2 + (ys - cy)**2)
idx = np.argmax(dists)
tip_x, tip_y = coords[idx]

print("Centroid:", (cx, cy))
print("TIP ESTIMATE:", (int(tip_x), int(tip_y)))

# --- 6. Draw results for visual testing ---
vis = after.copy()
cv2.circle(vis, (int(cx), int(cy)), 8, (0,255,0), -1)    # green = centroid
cv2.circle(vis, (int(tip_x), int(tip_y)), 8, (0,0,255), -1)  # red = estimated tip

cv2.imwrite("tip_debug.jpg", vis)
print("Saved tip_debug.jpg")