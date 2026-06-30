import cv2
import cv2.aruco as aruco
from vision.camera import Camera

cam = Camera("http://10.250.150.224:8081/")

# WICHTIG: IDs deiner Corner-Marker
CORNER_IDS = {
    "tl": 1,
    "tr": 2,
    "br": 3,
    "bl": 4
}

DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
detector = aruco.ArucoDetector(DICT)

def center(c):
    return tuple(map(int, c[0].mean(axis=0)))

while True:
    frame = cam.get_frame()
    if frame is None:
        continue

    corners, ids, _ = detector.detectMarkers(frame)

    if ids is None:
        cv2.putText(frame, "Keine Marker gefunden!", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    else:
        # Alle Marker anzeigen
        aruco.drawDetectedMarkers(frame, corners, ids)

        # Corner-Marker prüfen
        found = {}
        for c, i in zip(corners, ids):
            mid = center(c)
            marker_id = i[0]

            for name, cid in CORNER_IDS.items():
                if marker_id == cid:
                    found[name] = mid
                    cv2.circle(frame, mid, 10, (0,255,0), -1)
                    cv2.putText(frame, name, (mid[0]+10, mid[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # Debug: Zeige welche Corner fehlen
        missing = [name for name in CORNER_IDS if name not in found]
        if missing:
            cv2.putText(frame, f"Fehlend: {missing}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        else:
            cv2.putText(frame, "Alle Corner gefunden!", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Marker Test", frame)
    cv2.waitKey(1)
