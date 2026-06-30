
import cv2
import numpy as np

lower_blue = np.array([100, 80, 50])
upper_blue = np.array([140, 255, 255])

def detect_walls(topdown):
    hsv = cv2.cvtColor(topdown, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    mask = cv2.GaussianBlur(mask, (9, 9), 0)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)

    return mask
