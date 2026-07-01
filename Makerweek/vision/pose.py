import numpy as np
import math

# Marker-Kante 0→1 zeigt auf die LINKE Seite des AGV → +90° korrigiert auf Vorwärts
HEADING_OFFSET = math.pi / 2


class Pose:
    def __init__(self, x, y, theta):
        self.x     = x
        self.y     = y
        self.theta = theta


def compute_pose(marker_corners):
    """Berechnet Pose (x, y, theta) aus den 4 Ecken des ArUco-Markers."""
    if marker_corners is None:
        return None

    center = np.mean(marker_corners, axis=0)
    edge   = marker_corners[1] - marker_corners[0]
    theta  = math.atan2(edge[1], edge[0]) + HEADING_OFFSET

    return Pose(center[0], center[1], theta)


def world_to_grid(cx, cy, grid_size=40, W=800, H=800):
    """Pixel-Koordinaten → Gitterzelle (gx, gy)."""
    gx = int(cx / (W / grid_size))
    gy = int(cy / (H / grid_size))
    gx = max(0, min(grid_size - 1, gx))
    gy = max(0, min(grid_size - 1, gy))
    return gx, gy


def grid_to_world(gx, gy, grid_size=40, W=800, H=800):
    """Gitterzelle (gx, gy) → Pixel-Mittelpunkt der Zelle."""
    cx = (gx + 0.5) * (W / grid_size)
    cy = (gy + 0.5) * (H / grid_size)
    return cx, cy
