import cv2
import numpy as np

img = cv2.imread("warped_board.jpg")
clone = img.copy()
bullseye = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        bullseye.clear()
        bullseye.append((x, y))
        print(f"📍 Selected bullseye at: ({x}, {y})")

        # Mark + show
        marked = clone.copy()
        cv2.circle(marked, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Click bullseye center", marked)

        # Save to .npy for Pi
        np.save("bullseye_origin.npy", np.array([x, y]))
        print("💾 Saved as bullseye_origin.npy (copy this back to Pi)")

cv2.imshow("Click bullseye center", img)
cv2.setMouseCallback("Click bullseye center", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()