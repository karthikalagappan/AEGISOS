import cv2
import numpy as np
from module_4_alerts import process_alerts

# Create a black dummy image (simulating a camera frame)
dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

print("Test 1: Normal State")
frame_1 = process_alerts(dummy_frame.copy(), False, 0.1, "Standing")
cv2.imshow("Test 1 - Normal", frame_1)
cv2.waitKey(2000) # Show for 2 seconds

print("Test 2: Emergency State (Should Beep & Show Red Border)")
frame_2 = process_alerts(dummy_frame.copy(), True, 0.95, "Fallen")
cv2.imshow("Test 2 - Emergency", frame_2)
cv2.waitKey(0) # Wait until you press a key

cv2.destroyAllWindows()