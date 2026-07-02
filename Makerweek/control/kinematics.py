import math
import os
import json
import vision.pose as vp
from vision.pose import grid_to_world

# ============================================================
# Kalibrierungskonstanten — nach TestCalibrate.py anpassen!
# ============================================================
STEPS_PER_RAD      = 71.6   # Steps pro Radian (gemessen V3: 180°=225 Steps → 225/π)
DRIVE_VELOCITY     = 8      # Fahrgeschwindigkeit % (gemessener Live-Wert V3)
TURN_VELOCITY_PERC = 8      # Drehgeschwindigkeit % (gemessener Live-Wert V3)
ACCEL_PERC         = 3      # Beschleunigung % (gemessener Live-Wert V3)
CELL_PX            = 800 / 40          # 20 px pro Gitterzelle

# ── Feld-Geometrie (NICHT quadratisch!) ─────────────────────────────────────────
# Feld 200×300 cm wird auf 800×800 px gewarpt (PADDING=50 → 700 px Feld-Spanne).
# Dadurch ist der Maßstab je Achse verschieden → Steps/Pixel für x ≠ y!
STEPS_PER_CM   = 6.85               # gemessen V3
FIELD_PX       = 800 - 2 * 50       # Feld-Spanne im Top-Down = 700 px
FIELD_W_CM     = 200                # Breite  (Bild horizontal, tl→tr, links↔rechts)
FIELD_H_CM     = 300                # Höhe    (Bild vertikal, oben→unten = Fahrtrichtung)
# Steps pro Top-Down-Pixel je Achse (Startwerte, werden selbst kalibriert)
STEPS_PER_PX_X = STEPS_PER_CM * FIELD_W_CM / FIELD_PX   # ~1.96 (x = 200 cm)
STEPS_PER_PX_Y = STEPS_PER_CM * FIELD_H_CM / FIELD_PX   # ~2.94 (y = 300 cm)

# Ausrichtung: erst drehen bis Fehler < TURN_ALIGN_DEG, DANN fahren.
# So kann das AGV nie „mitten in der Kurve" losfahren und Wände schneiden.
TURN_ALIGN_DEG     = 12     # innerhalb dieser Toleranz gilt als ausgerichtet
MAX_TURN_STEP_DEG  = 55     # max Drehung pro Kommando (verhindert Überschwingen)
TURN_KP            = 0.7    # Dämpfung: pro Tick nur Bruchteil des Fehlers drehen
                            #           → konvergiert, statt zu überschwingen

# Manhattan-Fahrt: gerade Strecke (mehrere gleich gerichtete Zellen) am Stück
# fahren, dann stoppen + Kamera-Feedback, dann 90° drehen, usw.
# Kappung begrenzt die Drift bei langen Geraden → danach frisches Feedback.
MAX_RUN_CELLS      = 4      # max Zellen pro gerader Fahrt, dann Kamera-Feedback

# Drehrichtung: drive(steps, -steps).
# Wird AUTOMATISCH kalibriert: dreht sich das AGV vom Ziel WEG (Fehler wächst),
# kippt TURN_SIGN von selbst. Manuell per Terminal-'i' umkehrbar.
TURN_SIGN          = 1

# Standardwerte (für Reset) — die obigen Werte werden zur Laufzeit selbst kalibriert
_DEFAULT_STEPS_PER_RAD  = STEPS_PER_RAD
_DEFAULT_STEPS_PER_PX_X = STEPS_PER_PX_X
_DEFAULT_STEPS_PER_PX_Y = STEPS_PER_PX_Y


def _steps_for_drive(dxp, dyp):
    """Steps für eine gerade Fahrt um (dxp,dyp) Pixel — je nach dominanter Achse
    (Feld ist nicht quadratisch, x und y haben verschiedene Steps/Pixel)."""
    if abs(dxp) >= abs(dyp):
        return abs(dxp) * STEPS_PER_PX_X, "x"
    return abs(dyp) * STEPS_PER_PX_Y, "y"

# Nicht mehr benutzt, aber für alte Testskripte erhalten:
LATERAL_CORR_DEG   = 0.0
LOOKAHEAD_CELLS    = 2
TURN_GAIN          = 1.0
MIN_TURN_RAD       = math.radians(5)

# ── Selbstkalibrierung ────────────────────────────────────────────────────────
# Aus jeder Drehung/Fahrt wird gemessen: kommandierte Steps vs. tatsächlich von der
# Kamera gemessene Winkel-/Wegänderung → STEPS_PER_RAD / STEPS_PER_PX werden per
# gleitendem Mittel angepasst und in calibration.json gespeichert (überlebt Neustart).
CALIB_FILE   = os.path.join(os.path.dirname(__file__), "..", "calibration.json")
CALIB_ALPHA  = 0.25    # Gewicht jeder neuen Messung (höher = schneller, unruhiger)
CALIB_ON     = True    # Selbstkalibrierung aktiv?

# HEADING_OFFSET (vision/pose.py) wird aus der tatsächlichen Fahrtrichtung nachgeführt:
# fährt das AGV geradeaus, ist die gemessene Bewegungsrichtung die echte Vorwärts-
# richtung. Abweichung zur gemeldeten Heading = Offset-Fehler → sanft korrigieren.
HEAD_ALPHA     = 0.25              # EMA-Gewicht für Heading-Korrektur
HEAD_DEADBAND  = math.radians(4)   # kleiner Fehler = Rauschen, ignorieren
HEAD_MAX_STEP  = math.radians(12)  # max Korrektur pro Fahrt (gegen Ausreißer)
_DEFAULT_HEADING_OFFSET = vp.HEADING_OFFSET
# ============================================================


# Zustand für automatische Drehrichtungs-Erkennung
_last_turn_error = None   # Fehler der letzten Drehung (rad)
_turn_grew_count = 0      # wie oft der Fehler nacheinander GRÖSSER wurde
_pending_calib   = None   # zuletzt kommandierte Bewegung, wartet auf Kamera-Messung


def flip_turn_sign():
    """Kehrt die Drehrichtung um (Terminal-Befehl 'i')."""
    global TURN_SIGN
    TURN_SIGN = -TURN_SIGN
    print(f"[KALIB] TURN_SIGN = {TURN_SIGN:+d}  (Drehrichtung umgekehrt)")
    _reset_turn_tracking()
    save_calibration()
    return TURN_SIGN


def _reset_turn_tracking():
    """Setzt die Auto-Erkennung zurück (nach Fahren oder Ziel-Wechsel)."""
    global _last_turn_error, _turn_grew_count
    _last_turn_error = None
    _turn_grew_count = 0


def _calib_summary():
    return (f"STEPS_PER_RAD={STEPS_PER_RAD:.0f}  "
            f"STEPS_PER_PX x={STEPS_PER_PX_X:.2f}/y={STEPS_PER_PX_Y:.2f}  "
            f"TURN_SIGN={TURN_SIGN:+d}  HEADING={math.degrees(vp.HEADING_OFFSET):.0f}°")


def load_calibration():
    """Lädt gespeicherte Kalibrierung (falls vorhanden) beim Start."""
    global STEPS_PER_RAD, STEPS_PER_PX_X, STEPS_PER_PX_Y, TURN_SIGN
    try:
        with open(CALIB_FILE) as f:
            d = json.load(f)
        STEPS_PER_RAD     = float(d.get("steps_per_rad",  STEPS_PER_RAD))
        STEPS_PER_PX_X    = float(d.get("steps_per_px_x", STEPS_PER_PX_X))
        STEPS_PER_PX_Y    = float(d.get("steps_per_px_y", STEPS_PER_PX_Y))
        TURN_SIGN         = int(d.get("turn_sign",        TURN_SIGN))
        vp.HEADING_OFFSET = float(d.get("heading_offset", vp.HEADING_OFFSET))
        print(f"[CALIB] geladen: {_calib_summary()}")
    except (FileNotFoundError, ValueError, OSError):
        print("[CALIB] keine Datei – Standardwerte, kalibriert sich beim Fahren selbst")


def save_calibration():
    """Speichert die aktuelle Kalibrierung für den nächsten Start."""
    try:
        with open(CALIB_FILE, "w") as f:
            json.dump({"steps_per_rad":  round(STEPS_PER_RAD, 1),
                       "steps_per_px_x": round(STEPS_PER_PX_X, 3),
                       "steps_per_px_y": round(STEPS_PER_PX_Y, 3),
                       "turn_sign":      TURN_SIGN,
                       "heading_offset": round(vp.HEADING_OFFSET, 4)}, f, indent=2)
    except OSError as e:
        print(f"[CALIB] Speichern fehlgeschlagen: {e}")


def reset_calibration():
    """Setzt Kalibrierung auf Standardwerte zurück (Terminal-Befehl 'c')."""
    global STEPS_PER_RAD, STEPS_PER_PX_X, STEPS_PER_PX_Y, TURN_SIGN, _pending_calib
    STEPS_PER_RAD     = _DEFAULT_STEPS_PER_RAD
    STEPS_PER_PX_X    = _DEFAULT_STEPS_PER_PX_X
    STEPS_PER_PX_Y    = _DEFAULT_STEPS_PER_PX_Y
    TURN_SIGN         = 1
    vp.HEADING_OFFSET = _DEFAULT_HEADING_OFFSET
    _pending_calib    = None
    _reset_turn_tracking()
    save_calibration()
    print(f"[CALIB] zurückgesetzt: {_calib_summary()}")


def cancel_calibration():
    """Verwirft eine ausstehende Messung (z.B. Marker während der Fahrt verloren)."""
    global _pending_calib
    _pending_calib = None


def _apply_calibration(pose, pose_fresh):
    """Misst die zuvor kommandierte Bewegung (Kamera) und passt die Konstanten an.

    Kalibriert NUR wenn sowohl die Pose VOR dem Zug als auch die jetzige Pose echte
    Kamera-Messungen sind (fresh) — im Blind-/Dead-Reckoning-Betrieb wären die
    Messungen sonst wertlos und würden gute Werte verderben.
    """
    global STEPS_PER_RAD, STEPS_PER_PX_X, STEPS_PER_PX_Y, _pending_calib
    if _pending_calib is None or not CALIB_ON or pose is None:
        return
    fresh0 = _pending_calib[-1]
    if not (fresh0 and pose_fresh):
        _pending_calib = None      # gemischte/geschätzte Pose → Messung verwerfen
        return
    kind    = _pending_calib[0]
    changed = False

    if kind == "turn":
        _, theta0, steps, _ = _pending_calib
        dtheta = abs(_normalize_angle(pose.theta - theta0))
        # nur wenn spürbar gedreht und nicht am Wrap-Rand (>120° = mehrdeutig)
        if math.radians(6) < dtheta < math.radians(120) and abs(steps) > 3:
            measured      = max(20.0, min(300.0, abs(steps) / dtheta))
            STEPS_PER_RAD = (1 - CALIB_ALPHA) * STEPS_PER_RAD + CALIB_ALPHA * measured
            print(f"[CALIB] Drehung {math.degrees(dtheta):.0f}° bei {abs(steps)} steps "
                  f"→ STEPS_PER_RAD={STEPS_PER_RAD:.0f}")
            changed = True

    elif kind == "drive":
        _, x0, y0, theta0, steps, _ = _pending_calib
        dxp, dyp = pose.x - x0, pose.y - y0
        dist = math.hypot(dxp, dyp)
        # nur wenn spürbar gefahren und plausibel (kein Marker-Sprung)
        if 8 < dist < 250 and steps > 3:
            # Steps/Pixel je nach dominanter Achse kalibrieren (Feld nicht quadratisch)
            if abs(dxp) >= abs(dyp):
                measured       = max(0.5, min(8.0, steps / abs(dxp)))
                STEPS_PER_PX_X = (1 - CALIB_ALPHA) * STEPS_PER_PX_X + CALIB_ALPHA * measured
                axis = "x"
            else:
                measured       = max(0.5, min(8.0, steps / abs(dyp)))
                STEPS_PER_PX_Y = (1 - CALIB_ALPHA) * STEPS_PER_PX_Y + CALIB_ALPHA * measured
                axis = "y"
            print(f"[CALIB] Fahrt {dist:.0f}px ({axis}) bei {steps} steps  "
                  f"→ STEPS_PER_PX x={STEPS_PER_PX_X:.2f}/y={STEPS_PER_PX_Y:.2f}")
            changed = True
            # HEADING_OFFSET nachführen: echte Bewegungsrichtung vs. gemeldete Heading.
            if dist > 20:
                head_err = _normalize_angle(math.atan2(dyp, dxp) - theta0)
                if abs(head_err) > HEAD_DEADBAND:
                    corr = max(-HEAD_MAX_STEP, min(HEAD_MAX_STEP, head_err))
                    vp.HEADING_OFFSET = _normalize_angle(
                        vp.HEADING_OFFSET + HEAD_ALPHA * corr)
                    print(f"[CALIB] Fahrtrichtung {math.degrees(head_err):+.0f}° schräg "
                          f"→ HEADING_OFFSET={math.degrees(vp.HEADING_OFFSET):+.0f}°")

    _pending_calib = None
    if changed:
        save_calibration()


def _normalize_angle(a):
    return ((a + math.pi) % (2 * math.pi)) - math.pi


def fix_path_start_direction(path, pose, max_turn_deg=100.0):
    """(Ungenutzt) Überspringt rückwärtige Wegpunkte am Pfadanfang."""
    if len(path) < 2:
        return path
    max_rad = math.radians(max_turn_deg)
    for i in range(1, len(path)):
        wp = grid_to_world(path[i][0], path[i][1])
        dx, dy = wp[0] - pose.x, wp[1] - pose.y
        if math.hypot(dx, dy) < 2.0:
            continue
        err = abs(_normalize_angle(math.atan2(dy, dx) - pose.theta))
        if err <= max_rad:
            return [path[0]] + path[i:]
    return path


def turn_to_angle(agv, pose, waypoint_pixel, extra_angle=0.0, pose_fresh=True):
    """Dreht gedämpft (TURN_KP) Richtung Wegpunkt und kalibriert die Drehrichtung.

    Dreht pro Aufruf nur TURN_KP·Fehler (geklemmt) → konvergiert über mehrere
    Drehungen mit frischer Kamera-Pose. Die AUTO-DREHRICHTUNG (TURN_SIGN kippt wenn
    der Fehler wächst) wird nur mit ECHTER Kamera-Pose ausgewertet (pose_fresh).
    Gibt die dead-reckon-geschätzte neue Heading (rad) zurück.
    """
    global TURN_SIGN, _last_turn_error, _turn_grew_count, _pending_calib

    dx = waypoint_pixel[0] - pose.x
    dy = waypoint_pixel[1] - pose.y
    target_angle = math.atan2(dy, dx) + extra_angle
    error = _normalize_angle(target_angle - pose.theta)

    # Auto-Drehrichtungs-Erkennung nur mit echter Kamera-Pose (sonst wertlos)
    auto = ""
    if pose_fresh:
        flipped = False
        if _last_turn_error is not None:
            grew = abs(error) - abs(_last_turn_error)
            if grew > math.radians(10):
                TURN_SIGN = -TURN_SIGN
                _turn_grew_count = 0
                flipped = True
            elif grew > math.radians(2):
                _turn_grew_count += 1
                if _turn_grew_count >= 2:
                    TURN_SIGN = -TURN_SIGN
                    _turn_grew_count = 0
                    flipped = True
            else:
                _turn_grew_count = 0
        if flipped:
            auto = f"  [AUTO: falsch herum → TURN_SIGN={TURN_SIGN:+d}]"
            save_calibration()
        _last_turn_error = error

    max_rad = math.radians(MAX_TURN_STEP_DEG)
    cmd     = max(-max_rad, min(max_rad, error * TURN_KP))   # gedämpft + geklemmt
    steps   = int(TURN_SIGN * cmd * STEPS_PER_RAD)

    print(f"[TURN] heading={math.degrees(pose.theta):+.0f}°  "
          f"target={math.degrees(target_angle):+.0f}°  "
          f"error={math.degrees(error):+.0f}°  →  dreh {math.degrees(cmd):+.0f}° "
          f"steps={steps} (SIGN={TURN_SIGN:+d}){auto}")

    if steps != 0:
        agv.set_max_acceleration(ACCEL_PERC)
        agv.set_max_velocity(TURN_VELOCITY_PERC)
        # drive(steps, -steps): Vorzeichen = Drehrichtung (per TURN_SIGN kalibrierbar)
        agv.drive(steps, -steps)
        agv.wait_for_stop()
        # Messung für Selbstkalibrierung merken (theta VOR der Drehung + Frische-Flag)
        _pending_calib = ("turn", pose.theta, steps, pose_fresh)
    # Dead-Reckoning: geschätzte neue Heading (Drehung um cmd in Richtung Ziel)
    return _normalize_angle(pose.theta + cmd)


def _straight_run_end(path):
    """Länge der geraden Strecke ab path[0]: bündelt aufeinanderfolgende Zellen
    gleicher Richtung (max MAX_RUN_CELLS). Gibt den Index des Streckenendes."""
    dx = path[1][0] - path[0][0]
    dy = path[1][1] - path[0][1]
    run_end = 1
    while (run_end < len(path) - 1
           and run_end < MAX_RUN_CELLS
           and (path[run_end + 1][0] - path[run_end][0]) == dx
           and (path[run_end + 1][1] - path[run_end][1]) == dy):
        run_end += 1
    return run_end


def needs_turn(pose, path):
    """True wenn der nächste Zug eine Drehung ist (AGV nicht auf Streckenrichtung
    ausgerichtet). Main nutzt das, um VOR Drehungen auf ein frisches Kamerabild zu
    warten (Genauigkeit an Ecken), Geraden aber blind zu fahren."""
    if pose is None or path is None or len(path) < 2:
        return False
    run_end = _straight_run_end(path)
    tx, ty  = grid_to_world(path[run_end][0], path[run_end][1])
    error   = _normalize_angle(math.atan2(ty - pose.y, tx - pose.x) - pose.theta)
    return abs(error) > math.radians(TURN_ALIGN_DEG)


def execute_next_segment(agv, pose, path, grid=None, max_vel=DRIVE_VELOCITY,
                         pose_fresh=True):
    """Ein Manhattan-Zug: ENTWEDER auf die Streckenrichtung drehen ODER die gerade
    Strecke bis zur Ecke am Stück fahren. Fahrstrecke wird pro Achse in Steps
    umgerechnet (Feld nicht quadratisch).

    Rückgabe (count, target_pixel, est_pose):
      count = 0        → nur gedreht (keine Zelle konsumiert)
      count = run_end  → gerade Strecke gefahren (run_end Zellen konsumiert)
      est_pose         → Dead-Reckoning-Schätzung der Pose NACH dem Zug
    """
    global _pending_calib

    # Zuerst den zuvor kommandierten Zug vermessen (nur wenn Posen echt/fresh sind)
    _apply_calibration(pose, pose_fresh)

    run_end      = _straight_run_end(path)
    target_pixel = grid_to_world(path[run_end][0], path[run_end][1])
    dx = target_pixel[0] - pose.x
    dy = target_pixel[1] - pose.y
    dist_px = math.hypot(dx, dy)
    error   = _normalize_angle(math.atan2(dy, dx) - pose.theta)

    print(f"[SEG]  pos=({pose.x:.0f},{pose.y:.0f})  strecke→grid{tuple(path[run_end])}  "
          f"({run_end} Zellen)  dist={dist_px:.0f}px  error={math.degrees(error):+.0f}°"
          f"  {'[Kamera]' if pose_fresh else '[geschätzt]'}")

    # Sicherheit: Ziel liegt HINTER dem AGV und ist nah → überschossen → überspringen.
    if abs(error) > math.radians(115) and dist_px < CELL_PX * 1.4:
        print("[SKIP] Ziel liegt dicht hinter dem AGV (überschossen) → überspringe")
        _reset_turn_tracking()
        return 1, target_pixel, pose

    # Noch nicht ausgerichtet → NUR drehen (90°-Ecke).
    if abs(error) > math.radians(TURN_ALIGN_DEG):
        new_theta = turn_to_angle(agv, pose, target_pixel, pose_fresh=pose_fresh)
        return 0, target_pixel, vp.Pose(pose.x, pose.y, new_theta)

    # Ausgerichtet → gerade Strecke fahren. Steps je nach Achse (Feld nicht quadratisch).
    _reset_turn_tracking()
    drive_dist   = min(dist_px, CELL_PX * (MAX_RUN_CELLS + 0.5))
    scale        = drive_dist / dist_px if dist_px > 1e-6 else 0.0
    ddx, ddy     = dx * scale, dy * scale     # tatsächlich gefahrener px-Vektor
    steps_f, axis = _steps_for_drive(ddx, ddy)
    drive_steps  = max(1, int(steps_f))

    print(f"[DRIVE] {run_end} Zellen gerade ({axis})  →  {drive_steps} Steps "
          f"({drive_dist:.0f}px)  vel={max_vel}%")
    agv.set_max_acceleration(ACCEL_PERC)
    agv.set_max_velocity(max_vel)
    agv.drive(drive_steps, drive_steps)
    # Messung merken (Position + Heading + Frische VOR der Fahrt)
    _pending_calib = ("drive", pose.x, pose.y, pose.theta, drive_steps, pose_fresh)
    # Dead-Reckoning: geschätzte neue Position = angefahrenes Streckenende
    est = vp.Pose(pose.x + ddx, pose.y + ddy, pose.theta)
    return run_end, target_pixel, est


# Gespeicherte Kalibrierung beim Import laden (falls calibration.json existiert)
load_calibration()
