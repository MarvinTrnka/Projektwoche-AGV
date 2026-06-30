import numpy as np
import math

class Pose:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

def compute_pose(marker_corners):
    if marker_corners is None:
        return None

    center = np.mean(marker_corners, axis=0)

    # Richtung: Kante zwischen Ecke 0 und 1
    edge = marker_corners[1] - marker_corners[0]
    theta = math.atan2(edge[1], edge[0])

    return Pose(center[0], center[1], theta)

def world_to_grid(cx, cy, grid_size=40, W=800, H=800):
    gx = int(cx / (W / grid_size))
    gy = int(cy / (H / grid_size))
    return gx, gy
