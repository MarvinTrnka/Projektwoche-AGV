import cv2
import cv2.aruco as aruco
import numpy as np

# !! WETTKAMPFTAG: IDs am Feld bestätigen, bevor Programm gestartet wird !!
CORNER_IDS = {
    "tl": 1,
    "tr": 2,
    "br": 3,
    "bl": 4,
}

DICT    = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
_params = aruco.DetectorParameters()
_params.minMarkerPerimeterRate      = 0.01   # sehr kleine/weite Marker erkennen
_params.adaptiveThreshWinSizeStep   = 4      # mehr Schwellwert-Fenster = robuster
_params.polygonalApproxAccuracyRate = 0.08   # leicht verzerrte Marker erlaubt
_params.cornerRefinementMethod      = aruco.CORNER_REFINE_SUBPIX
detector = aruco.ArucoDetector(DICT, _params)

AGV_MARKER_ID = 7   # vom Lehrer zugewiesen; ggf. anpassen!

last_corners = {}

_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

def _enhance(frame):
    """Kontrastverstärkung für schlechte Beleuchtung."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(_clahe.apply(gray), cv2.COLOR_GRAY2BGR)

def _warp_pts(pts4x2, warp_matrix):
    pts_h = pts4x2.reshape(-1, 1, 2).astype(np.float32)
    pts_w = cv2.perspectiveTransform(pts_h, warp_matrix)
    return pts_w.reshape(4, 2)

def _center(c):
    return np.mean(c[0], axis=0)


def detect_corner_markers(frame):
    global last_corners
    corners, ids, _ = detector.detectMarkers(frame)
    if ids is not None:
        for c, i in zip(corners, ids):
            for name, cid in CORNER_IDS.items():
                if i[0] == cid:
                    last_corners[name] = _center(c)
    return last_corners if len(last_corners) == 4 else None


def _find_agv(corners, ids, warp_matrix, topdown_coords=False):
    """Sucht AGV_MARKER_ID in erkannten Markern. Gibt Ecken zurück oder None."""
    if ids is None:
        return None
    for c, i in zip(corners, ids):
        if int(i[0]) == AGV_MARKER_ID:
            pts = c[0]
            if topdown_coords:
                return pts  # bereits in topdown-Koordinaten
            return _warp_pts(pts, warp_matrix) if warp_matrix is not None else pts
    return None


def detect_markers(frame, warp_matrix, topdown, stale_frames=0):
    """Erkennt AGV-Marker (ID 7) + fremde Marker in 3 Aufrufen.

    Strategie:
      1. Rohbild (640×480) — findet Marker auch außerhalb des Felds
      2. Topdown (800×800) — findet Marker auf dem Feld, holt fremde AGVs
      3. CLAHE-verstärktes Rohbild — immer als Fallback für schwaches Licht
    """
    known_corner_ids = set(CORNER_IDS.values())
    foreign = []

    # ── Schuss 1: Rohbild ──────────────────────────────────────────────────────
    c1, ids1, _ = detector.detectMarkers(frame)
    agv_corners  = _find_agv(c1, ids1, warp_matrix, topdown_coords=False)

    # ── Schuss 2: Topdown ──────────────────────────────────────────────────────
    c2, ids2, _ = detector.detectMarkers(topdown)
    if agv_corners is None:
        agv_corners = _find_agv(c2, ids2, warp_matrix, topdown_coords=True)
    # Fremde AGVs immer aus Topdown (topdown-Koordinaten für Visualisierung)
    if ids2 is not None:
        for c, i in zip(c2, ids2):
            mid = int(i[0])
            if mid not in known_corner_ids and mid != AGV_MARKER_ID:
                cx, cy = c[0].mean(axis=0)
                foreign.append((mid, (float(cx), float(cy))))

    # ── Schuss 3: CLAHE-Fallback ────────────────────────────────────────────────
    if agv_corners is None:
        enhanced    = _enhance(frame)
        c3, ids3, _ = detector.detectMarkers(enhanced)
        agv_corners = _find_agv(c3, ids3, warp_matrix, topdown_coords=False)

    return agv_corners, foreign
