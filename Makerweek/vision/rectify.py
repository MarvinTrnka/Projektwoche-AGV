import cv2
import numpy as np

# Erwartet ein Dictionary mit den vier Corner-Punkten:
# corners = {
#   "tl": (x,y),
#   "tr": (x,y),
#   "br": (x,y),
#   "bl": (x,y)
# }

PADDING = 50  # Pixel Rand außerhalb der Eckmarker → AGV hinter Startlinie sichtbar

def compute_warp_matrix(corners, W=800, H=800):
    src = np.float32([
        corners["tl"],
        corners["tr"],
        corners["br"],
        corners["bl"]
    ])

    dst = np.float32([
        [PADDING,       PADDING],
        [W - PADDING,   PADDING],
        [W - PADDING,   H - PADDING],
        [PADDING,       H - PADDING],
    ])

    return cv2.getPerspectiveTransform(src, dst)


def warp_topdown(frame, corners, W=800, H=800):
    """
    Erzeugt ein entzerrtes Top-Down-Bild basierend auf den Corner-Markern.
    """
    M = compute_warp_matrix(corners, W, H)
    topdown = cv2.warpPerspective(frame, M, (W, H))
    return topdown
