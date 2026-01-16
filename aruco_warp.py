import cv2
import cv2.aruco as aruco
import numpy as np

# Load the image
img = cv2.imread("after.jpg")

# Detect markers
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
params = aruco.DetectorParameters_create()
corners, ids, _ = aruco.detectMarkers(img, aruco_dict, parameters=params)

# Check we have all 4 markers
if ids is None or len(ids) != 4:
    print("⚠️ Could not find 4 markers — attempting to load saved warp matrix...")
    try:
        matrix = np.load("warp_matrix.npy")
        print("✅ Loaded saved warp matrix.")
    except FileNotFoundError:
        print("❌ No saved matrix found. Cannot proceed.")
        exit()

else:
    # Map marker IDs to their corners
    marker_map = {id[0]: corner for id, corner in zip(ids, corners)}

    # Define desired destination coordinates (e.g., 1600x848 rectangle)
    warp_width = 1600
    warp_height = 848
    dst_pts = np.array([
        [0, 0],                     # top-left
        [warp_width, 0],           # top-right
        [warp_width, warp_height], # bottom-right
        [0, warp_height],          # bottom-left
    ], dtype=np.float32)

    # Match source points using YOUR marker layout
    # Adjust this if your physical placement order changes
    src_pts = np.array([
        marker_map[0][0][0],  # top-left
        marker_map[1][0][1],  # top-right
        marker_map[3][0][2],  # bottom-right
        marker_map[2][0][3],  # bottom-left
    ], dtype=np.float32)

    # Compute warp matrix
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    np.save("warp_matrix.npy", matrix)
    print("💾 Saved warp matrix to warp_matrix.npy")

# Apply warp
warped = cv2.warpPerspective(img, matrix, (warp_width, warp_height))

# Save result
cv2.imwrite("warped_board.jpg", warped)
print("✅ Warp complete — saved to warped_board.jpg")