import numpy as np
import cv2

# detect all connected webcams
img = cv2.imread('./image.jpg')

cv2.imshow('webcam', img)
while True:
    k = cv2.waitKey(1)
    if k == ord('q') or k == 27:
        cv2.destroyAllWindows()
        break