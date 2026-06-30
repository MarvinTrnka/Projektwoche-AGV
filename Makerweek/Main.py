import numpy as np
import cv2


from vision.camera import Camera
from vision.markers import detect_corner_markers, detect_agv_marker
from vision.rectify import compute_warp, apply_warp
from vision.walls import detect_walls
from vision.pose import compute_pose

from planning.grid import build_grid
from planning.astar import astar

from control.kinematics import turn_to_angle, drive_distance
from control.agv_api import AGV

from utils.visualization import show_debug

def main():
    cam = Camera("http://10.250.150.224:8081/")
    agv = AGV("<agv-ip>")

    warp_matrix = None

    while True:
        frame = cam.get_frame()

        if frame is not None:
            cv2.imshow("Live Stream", frame)
            cv2.waitKey(1)

        # 1. Spielfeld-Ecken finden
        corners = detect_corner_markers(frame)

        # 2. Warp-Matrix einmal berechnen
        if warp_matrix is None and corners is not None:
            warp_matrix = compute_warp(corners)

        # 3. Bild entzerren
        topdown = apply_warp(frame, warp_matrix)

        # 4. Wände erkennen
        wall_mask = detect_walls(topdown)

        # 5. Grid bauen
        grid = build_grid(wall_mask)

        # 6. AGV-Pose bestimmen
        agv_marker = detect_agv_marker(topdown)
        pose = compute_pose(agv_marker)

        # 7. Pfad planen
        path = astar(grid, pose)

        # 8. Pfad abfahren
        for waypoint in path:
            turn_to_angle(agv, pose, waypoint)
            drive_distance(agv, pose, waypoint)
            frame = cam.get_frame()
            topdown = apply_warp(frame, warp_matrix)
            agv_marker = detect_agv_marker(topdown)
            pose = compute_pose(agv_marker)

        # Debug-Overlay anzeigen
        show_debug(topdown, grid, pose, path)

if __name__ == "__main__":
    main()
