import cv2
import numpy as np
import math

rotation_offset_deg = -7  # CCW correction

# Load warped image and bullseye origin
img = cv2.imread("warped_board.jpg")
origin = np.load("bullseye_origin.npy")
cx, cy = int(origin[0]), int(origin[1])

# Scoring ring radii (pixels) — these are estimates, adjust as needed
inner_bull = 10
outer_bull = 30
treble_inner = 100
treble_outer = 110
double_inner = 160
double_outer = 170

# Draw concentric rings
rings = [inner_bull, outer_bull, treble_inner, treble_outer, double_inner, double_outer]
for r in rings:
    cv2.circle(img, (cx, cy), r, (0, 255, 0), 1)

# Draw 20 sector lines (every 18°)
for i in range(20):
    angle_deg = i * 18 + rotation_offset_deg
    angle_rad = math.radians(angle_deg)
    x = int(cx + double_outer * math.cos(angle_rad))
    y = int(cy + double_outer * math.sin(angle_rad))
    cv2.line(img, (cx, cy), (x, y), (255, 0, 0), 1)

# Save overlay
cv2.imwrite("scoring_grid.jpg", img)
print("✅ Saved scoring grid to scoring_grid.jpg")