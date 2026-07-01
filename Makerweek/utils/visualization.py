import cv2
import math

CELL        = 800 // 40  # 20 px pro Gitterzelle
FINISH_Y    = 750        # y-Koordinate der Ziellinie (H - PADDING = 800 - 50)

# Status → (Text, BGR-Farbe)
_STATUS = {
    "no_marker":   ("AGV Marker (ID 7) nicht gefunden",   (0, 140, 255)),
    "no_path":     ("Kein Pfad gefunden!",                 (0, 140, 255)),
    "ready":       ("BEREIT  --  S druecken",              (0, 220,   0)),
    "starting":    ("Suche AGV – kriecht vor...",          (0, 220, 220)),
    "driving":     ("FAHREN!",                             (0, 255,   0)),
    "danger":      ("GEFAHR  --  Wand voraus!",             (0,   0, 180)),
    "pose_lost":   ("Marker verloren – fahrt langsam...",  (0,  80, 255)),
    "goal_reached":("*** ZIEL ERREICHT! ***",              (0, 255, 255)),
}


def show_debug(img, grid, pose, path, status="no_marker", goal=None,
               other_agvs=None):
    """
    Zeichnet Top-Down-Bild mit:
      - Roten Wandzellen
      - Cyan-Zielzelle
      - Grünem Pfad
      - Gelbem AGV-Kreis + Heading-Pfeil
      - Roten Kreisen für andere AGVs / Hindernisse
      - Status-Leiste oben
    """
    out = img.copy()

    # Geblockte Zellen rot
    if grid is not None:
        for gy in range(40):
            for gx in range(40):
                if grid[gy][gx] == 1:
                    x1, y1 = gx * CELL, gy * CELL
                    cv2.rectangle(out, (x1, y1), (x1 + CELL, y1 + CELL),
                                  (0, 0, 120), -1)

    # Zielzelle cyan
    if goal is not None:
        gx, gy = goal
        cv2.rectangle(out, (gx * CELL, gy * CELL),
                      (gx * CELL + CELL, gy * CELL + CELL), (255, 255, 0), -1)

    # Pfad grün
    if path is not None and len(path) > 1:
        for i in range(len(path) - 1):
            p1 = (int((path[i][0]     + 0.5) * CELL), int((path[i][1]     + 0.5) * CELL))
            p2 = (int((path[i + 1][0] + 0.5) * CELL), int((path[i + 1][1] + 0.5) * CELL))
            cv2.line(out, p1, p2, (0, 255, 0), 2)

    # Andere AGVs anzeigen (nur zur Info, NICHT als Hindernisse)
    if other_agvs:
        for mid, (px, py) in other_agvs:
            cx, cy = int(px), int(py)
            cv2.circle(out, (cx, cy), 18, (0, 80, 255), 3)
            cv2.putText(out, f"ID{mid}", (cx - 14, cy + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 80, 255), 2)

    # Ziellinie magenta
    cv2.line(out, (0, FINISH_Y), (800, FINISH_Y), (255, 0, 200), 2)
    cv2.putText(out, "ZIEL", (10, FINISH_Y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 200), 1)

    # AGV gelb + Heading-Pfeil magenta
    if pose is not None:
        cx, cy = int(pose.x), int(pose.y)
        cv2.circle(out, (cx, cy), 10, (0, 255, 255), -1)
        dx = int(30 * math.cos(pose.theta))
        dy = int(30 * math.sin(pose.theta))
        cv2.arrowedLine(out, (cx, cy), (cx + dx, cy + dy),
                        (255, 0, 255), 2, tipLength=0.4)

    # Status-Leiste
    text, color = _STATUS.get(status, ("?", (200, 200, 200)))
    bg_color = (0, 60, 0) if status == "goal_reached" else (0, 0, 0)
    cv2.rectangle(out, (0, 0), (800, 50), bg_color, -1)
    cv2.putText(out, text, (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    cv2.imshow("AGV Debug", out)
