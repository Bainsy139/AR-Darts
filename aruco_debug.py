import cv2
import cv2.aruco as aruco

# Load the image
img = cv2.imread("after.jpg")

# Setup detector
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
params = aruco.DetectorParameters_create()

# Detect
corners, ids, _ = aruco.detectMarkers(img, aruco_dict, parameters=params)

# Draw results
if ids is not None:
    img = aruco.drawDetectedMarkers(img, corners, ids)
    print("✅ Detected IDs:", ids.flatten())
else:
    print("❌ No markers detected.")

# Save output
cv2.imwrite("aruco_check.jpg", img)