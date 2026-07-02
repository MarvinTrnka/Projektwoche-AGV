import cv2
import numpy as np
from vision.camera import Camera
import cv2.aruco as aruco

cam = Camera("http://10.250.150.224:8081/")

DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
detector = aruco.ArucoDetector(DICT)

last_corners = {}

def center(c):
    return np.mean(c[0], axis=0)

def compute_warp(corners, W=800, H=800):
    src = np.float32([
        corners["tl"],
        corners["tr"],
        corners["br"],
        corners["bl"]
    ])

    dst = np.float32([
        [0, 0],
        [W, 0],
        [W, H],
        [0, H]
    ])

    return cv2.getPerspectiveTransform(src, dst)

while True:
    frame, _ = cam.get_frame()
    if frame is None:
        continue

    corners, ids, _ = detector.detectMarkers(frame)

    if ids is not None:
        for c, i in zip(corners, ids):
            mid = center(c)
            marker_id = i[0]

            if marker_id == 1: last_corners["tl"] = mid
            if marker_id == 2: last_corners["tr"] = mid
            if marker_id == 3: last_corners["br"] = mid
            if marker_id == 4: last_corners["bl"] = mid

    if len(last_corners) == 4:
        M = compute_warp(last_corners)
        topdown = cv2.warpPerspective(frame, M, (800, 800))

        cv2.imshow("Top Down", topdown)

    cv2.imshow("Original", frame)
    cv2.waitKey(1)
