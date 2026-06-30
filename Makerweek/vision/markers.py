import cv2
import cv2.aruco as aruco
import numpy as np

CORNER_IDS = {
    "tl": 1,
    "tr": 2,
    "br": 3,
    "bl": 4
}

DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
detector = aruco.ArucoDetector(DICT)

# Cache für stabile Marker
last_corners = {}

def center(c):
    return np.mean(c[0], axis=0)

def detect_corner_markers(frame):
    global last_corners

    corners, ids, _ = detector.detectMarkers(frame)

    if ids is not None:
        for c, i in zip(corners, ids):
            marker_id = i[0]
            for name, cid in CORNER_IDS.items():
                if marker_id == cid:
                    last_corners[name] = center(c)

    # Nur zurückgeben, wenn wir ALLE 4 Marker irgendwann gesehen haben
    if len(last_corners) == 4:
        return last_corners

    return None
