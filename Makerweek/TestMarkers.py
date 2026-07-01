import cv2
import cv2.aruco as aruco
import numpy as np
from vision.camera import Camera
from vision.markers import detect_corner_markers, AGV_MARKER_ID
from vision.rectify import compute_warp_matrix

cam = Camera("http://10.250.150.224:8081/")

DICT     = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
detector = aruco.ArucoDetector(DICT)

warp_matrix = None

while True:
    frame = cam.get_frame()
    if frame is None:
        continue

    # Eckmarker + Warp
    corners = detect_corner_markers(frame)
    if warp_matrix is None and corners is not None:
        warp_matrix = compute_warp_matrix(corners)

    # --- Rohbild: alle Marker einzeichnen ---
    raw = frame.copy()
    det_corners, ids, _ = detector.detectMarkers(frame)
    if ids is not None:
        aruco.drawDetectedMarkers(raw, det_corners, ids)
        for c, i in zip(det_corners, ids):
            mid = tuple(map(int, c[0].mean(axis=0)))
            color = (0, 255, 0) if i[0] == AGV_MARKER_ID else (255, 100, 0)
            cv2.putText(raw, f"ID:{i[0]}", (mid[0]-15, mid[1]-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    else:
        cv2.putText(raw, "KEINE MARKER", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    status = f"AGV-Marker ID={AGV_MARKER_ID} | Ecken: {'OK' if corners else 'fehlen'}"
    cv2.putText(raw, status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
    cv2.imshow("Rohbild — alle Marker", raw)

    # --- Top-Down: AGV-Marker suchen (im Rohbild, dann in Top-Down transformieren) ---
    if warp_matrix is not None:
        topdown = cv2.warpPerspective(frame, warp_matrix, (800, 800))
        td_out  = topdown.copy()

        # AGV im Rohbild suchen und Koordinaten transformieren
        from vision.markers import detect_agv_marker
        agv_corners = detect_agv_marker(frame, warp_matrix)
        if agv_corners is not None:
            mid = tuple(map(int, agv_corners.mean(axis=0)))
            pts = agv_corners.reshape(1, 4, 2).astype(int)
            cv2.polylines(td_out, pts, True, (0, 255, 0), 2)
            cv2.putText(td_out, f"AGV ID:{AGV_MARKER_ID}", (mid[0]-30, mid[1]-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            msg   = f"AGV (ID {AGV_MARKER_ID}) GEFUNDEN!"
            color = (0, 255, 0)
        else:
            msg   = f"AGV (ID {AGV_MARKER_ID}) NICHT GEFUNDEN"
            color = (0, 0, 255)

        cv2.putText(td_out, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("Top-Down — AGV Suche", td_out)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
