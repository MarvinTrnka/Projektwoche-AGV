import cv2
import numpy as np
from vision.camera import Camera

cam = Camera("http://10.250.150.224:8081/")

# HSV-Bereich für Blau
lower_blue = np.array([100, 80, 50])
upper_blue = np.array([140, 255, 255])

while True:
    frame, _ = cam.get_frame()
    if frame is None:
        continue

    # In HSV umwandeln
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Blau filtern
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Maske glätten
    mask = cv2.GaussianBlur(mask, (7, 7), 0)

    # Konturen finden
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Wände markieren
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300:  # kleine Flecken ignorieren
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # Rechteck zeichnen
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Koordinaten anzeigen
        cv2.putText(frame, f"Wall ({x},{y})",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 0, 0), 2)

    # Debug-Ausgabe
    cv2.imshow("Walls", frame)
    cv2.imshow("Mask", mask)
    cv2.waitKey(1)
