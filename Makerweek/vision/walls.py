import cv2
import numpy as np

lower_blue = np.array([95, 60, 40])
upper_blue = np.array([140, 255, 255])

def detect_walls(topdown):
    blur = cv2.GaussianBlur(topdown, (5, 5), 0)
    hsv  = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    k5 = np.ones((5, 5), np.uint8)
    # Lücken im Klebeband schließen
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5)

    # Kleine Noise-Blobs entfernen (< 80px = Staub/Schatten, kein echtes Klebeband)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= 80:
            clean[labels == label] = 255

    # KEIN Dilate mehr → Wände bleiben dünn, Korridore bleiben frei
    return clean
