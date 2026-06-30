import cv2
import numpy as np

from vision.camera import Camera
from vision.markers import detect_corner_markers
from vision.rectify import warp_topdown
from vision.walls import detect_walls
from vision.pose import world_to_grid
from planning.grid import mask_to_grid
from planning.astar import astar

cam = Camera("http://10.250.150.224:8081/")

# Beispiel-Ziel im Grid
goal = (10, 5)

def draw_path(topdown, path, grid_size=40):
    H, W, _ = topdown.shape
    cell_h = H // grid_size
    cell_w = W // grid_size

    for gx, gy in path:
        x1 = gx * cell_w
        y1 = gy * cell_h
        x2 = x1 + cell_w
        y2 = y1 + cell_h
        cv2.rectangle(topdown, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return topdown


while True:
    frame = cam.get_frame()
    if frame is None:
        continue

    # --- Corner Marker erkennen ---
    corners = detect_corner_markers(frame)

    if corners is None:
        cv2.putText(frame, "Warte auf Corner Marker...", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow("Test", frame)
        cv2.waitKey(1)
        continue

    # --- TopDown erzeugen ---
    topdown = warp_topdown(frame, corners)

    # --- Wände erkennen ---
    mask = detect_walls(topdown)

    # --- Grid erzeugen ---
    grid = mask_to_grid(mask)

    # --- AGV Position (TEST: Mitte unten) ---
    agv_pos = (400, 700)
    agv_grid = world_to_grid(*agv_pos)

    # --- Pfad berechnen ---
    path = astar(grid, agv_grid, goal)

    if path is None:
        cv2.putText(topdown, "Kein Pfad gefunden!", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    else:
        topdown = draw_path(topdown, path)

    # --- Anzeigen ---
    cv2.imshow("TopDown Path", topdown)
    cv2.imshow("Mask", mask)
    cv2.waitKey(1)
