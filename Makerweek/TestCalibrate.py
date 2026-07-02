"""
TestCalibrate.py — Kamerabasierte Kalibrierung
===============================================
Misst STEPS_PER_CELL und STEPS_PER_RAD automatisch mit der Kamera.
Mehrere Messungen werden gemittelt für höhere Genauigkeit.

Ausführen aus dem Makerweek/ Verzeichnis:
  python TestCalibrate.py

Tasten (CV2-Fenster fokussiert):
  d  → Vorwärtsfahrt DRIVE_STEPS Steps  → misst STEPS_PER_CELL
  r  → Rechtsdrehung TURN_STEPS Steps   → misst STEPS_PER_RAD
  l  → Linksdrehung  TURN_STEPS Steps   → misst STEPS_PER_RAD
  +  → DRIVE_STEPS erhöhen (+50)
  -  → DRIVE_STEPS verringern (-50)
  q  → Beenden + Zusammenfassung ausgeben

Ablauf pro Messung:
  1. Startpose aus 10 Kamera-Frames mitteln
  2. AGV fahren/drehen lassen
  3. Endpose aus 10 Kamera-Frames mitteln
  4. Differenz berechnen → Kalibrierwert ausgeben
"""
import cv2
import math
import time

from vision.camera import Camera
from vision.markers import detect_corner_markers, detect_markers
from vision.rectify import compute_warp_matrix
from vision.pose import compute_pose
from control.agv_api import AGV

# ─── Konfiguration ────────────────────────────────────────────────────────────
CAMERA_URL  = "http://10.250.150.224:8081/"
AGV_IP      = "172.17.1.73"
DRIVE_STEPS = 400    # Steps vorwärts (≈ 2 Zellen bei korrekter Kalibrierung)
TURN_STEPS  = 130    # Steps für Drehtest (≈ 90° bei korrekter Kalibrierung)
VEL_DRIVE   = 20    # % Fahrgeschwindigkeit — höher als Main.py damit Motoren sicher anlaufen
VEL_TURN    = 20    # % Drehgeschwindigkeit
CELL_PX     = 800 / 40   # 20 px pro Gitterzelle
# ─────────────────────────────────────────────────────────────────────────────


def _stable_pose(cam, warp_matrix, n=10):
    """Mittelt n gemessene Poses für höhere Genauigkeit (Kreismittelwert für theta)."""
    poses = []
    deadline = time.time() + 5.0
    while len(poses) < n and time.time() < deadline:
        frame, _ = cam.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue
        topdown = cv2.warpPerspective(frame, warp_matrix, (800, 800))
        marker, _ = detect_markers(frame, warp_matrix, topdown)
        pose = compute_pose(marker)
        if pose is not None:
            poses.append(pose)
        time.sleep(0.05)

    if not poses:
        return None

    avg_x = sum(p.x for p in poses) / len(poses)
    avg_y = sum(p.y for p in poses) / len(poses)
    # Kreismittelwert für Winkel (korrekt bei Übergang ±π)
    avg_t = math.atan2(
        sum(math.sin(p.theta) for p in poses) / len(poses),
        sum(math.cos(p.theta) for p in poses) / len(poses),
    )

    class _Pose:
        x, y, theta = avg_x, avg_y, avg_t
    return _Pose()


def main():
    global DRIVE_STEPS

    cam = Camera(CAMERA_URL)
    agv = AGV(AGV_IP)

    warp_matrix  = None
    cell_results = []   # gemessene STEPS_PER_CELL Werte
    rad_results  = []   # gemessene STEPS_PER_RAD Werte

    print("=" * 55)
    print("TestCalibrate — kamerabasierte Kalibrierung")
    print("=" * 55)
    print("Warte auf Eckmarker (IDs 1–4)...")

    # ── Spielfeld + Warp erkennen ──────────────────────────────────
    while warp_matrix is None:
        frame, _ = cam.get_frame()
        if frame is None:
            time.sleep(0.1)
            continue
        corners = detect_corner_markers(frame)
        if corners:
            warp_matrix = compute_warp_matrix(corners)
            agv.enable()
            print("Spielfeld erkannt!  Motoren ein.")
            print()
            print(f"  d  → {DRIVE_STEPS} Steps vor → STEPS_PER_CELL")
            print(f"  r  → {TURN_STEPS}  Steps rechts drehen → STEPS_PER_RAD")
            print(f"  l  → {TURN_STEPS}  Steps links  drehen → STEPS_PER_RAD")
            print("  +/- → DRIVE_STEPS ±50")
            print("  q  → Beenden")
        display = (cv2.warpPerspective(frame, warp_matrix, (800, 800))
                   if warp_matrix is not None else cv2.resize(frame, (800, 800)))
        cv2.imshow("TestCalibrate", display)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            return

    # ── Haupt-Loop ─────────────────────────────────────────────────
    while True:
        frame, _ = cam.get_frame()
        if frame is None:
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
            continue

        topdown = cv2.warpPerspective(frame, warp_matrix, (800, 800))
        marker, _ = detect_markers(frame, warp_matrix, topdown)
        pose = compute_pose(marker)

        # Live-Anzeige
        display = topdown.copy()
        if pose is not None:
            cx, cy = int(pose.x), int(pose.y)
            dx = int(30 * math.cos(pose.theta))
            dy = int(30 * math.sin(pose.theta))
            cv2.circle(display, (cx, cy), 10, (0, 255, 255), -1)
            cv2.arrowedLine(display, (cx, cy), (cx + dx, cy + dy),
                            (255, 0, 255), 2, tipLength=0.4)
            cv2.putText(display,
                        f"({cx}, {cy})   {math.degrees(pose.theta):.1f}deg",
                        (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        else:
            cv2.putText(display, "Kein Marker (ID 7)!", (10, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Laufende Durchschnittswerte einblenden
        y = 70
        if cell_results:
            avg = sum(cell_results) / len(cell_results)
            cv2.putText(display,
                        f"STEPS_PER_CELL ~ {avg:.0f}  (n={len(cell_results)})",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
            y += 30
        if rad_results:
            avg = sum(rad_results) / len(rad_results)
            cv2.putText(display,
                        f"STEPS_PER_RAD  ~ {avg:.0f}  (n={len(rad_results)})",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 180, 255), 2)

        cv2.putText(display, f"DRIVE_STEPS={DRIVE_STEPS}",
                    (10, 780), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
        cv2.imshow("TestCalibrate", display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            break

        # ── d: Fahrtest ───────────────────────────────────────────
        elif key == ord('d'):
            print(f"\n[d] Fahrtest: {DRIVE_STEPS} Steps, {VEL_DRIVE}% Geschwindigkeit")

            start = _stable_pose(cam, warp_matrix)
            if start is None:
                print("  FEHLER: Startpose nicht erkannt!")
                continue
            print(f"  Start: ({start.x:.1f}, {start.y:.1f})  "
                  f"theta={math.degrees(start.theta):.1f}deg")

            agv.set_max_acceleration(25)
            agv.set_max_velocity(VEL_DRIVE)
            agv.drive(DRIVE_STEPS, DRIVE_STEPS)
            agv.wait_for_stop(timeout=60)
            time.sleep(0.7)   # Kamera und Fahrzeug einpendeln

            end = _stable_pose(cam, warp_matrix)
            if end is None:
                print("  FEHLER: Endpose nach Fahrt nicht erkannt!")
                continue
            print(f"  Ende:  ({end.x:.1f}, {end.y:.1f})  "
                  f"theta={math.degrees(end.theta):.1f}deg")

            dist_px = math.hypot(end.x - start.x, end.y - start.y)
            cells   = dist_px / CELL_PX
            if cells < 0.1:
                print("  WARNUNG: Zu geringe Bewegung – Marker sichtbar?")
                continue

            spc = DRIVE_STEPS / cells
            cell_results.append(spc)
            avg = sum(cell_results) / len(cell_results)
            print(f"  Gefahren: {dist_px:.1f} px = {cells:.2f} Zellen")
            print(f"  >>> STEPS_PER_CELL = {spc:.0f}")
            print(f"  Durchschnitt ({len(cell_results)} Messung(en)): {avg:.0f}")

        # ── r / l: Drehtest ───────────────────────────────────────
        elif key in (ord('r'), ord('l')):
            sign = 1 if key == ord('r') else -1
            name = "Rechts" if sign == 1 else "Links"
            print(f"\n[{name[0].lower()}] Drehtest {name}: {TURN_STEPS} Steps")

            start = _stable_pose(cam, warp_matrix)
            if start is None:
                print("  FEHLER: Startpose nicht erkannt!")
                continue
            print(f"  Start-Winkel: {math.degrees(start.theta):.2f}deg")

            agv.set_max_acceleration(25)
            agv.set_max_velocity(VEL_TURN)
            # sign=+1 → drive(+, -) = Rechtsdrehung; sign=-1 → drive(-, +) = Linksdrehung
            agv.drive(sign * TURN_STEPS, -sign * TURN_STEPS)
            agv.wait_for_stop(timeout=30)
            time.sleep(0.7)

            end = _stable_pose(cam, warp_matrix)
            if end is None:
                print("  FEHLER: Endpose nach Drehung nicht erkannt!")
                continue
            print(f"  End-Winkel:   {math.degrees(end.theta):.2f}deg")

            # Winkel-Differenz (normalisiert auf [-π, π])
            delta = ((end.theta - start.theta) + math.pi) % (2 * math.pi) - math.pi
            delta_abs = abs(delta)
            if delta_abs < 0.02:
                print("  WARNUNG: Kein messbarer Winkel – Marker sichtbar?")
                continue

            spr = TURN_STEPS / delta_abs
            rad_results.append(spr)
            avg = sum(rad_results) / len(rad_results)
            print(f"  Winkel: {math.degrees(delta_abs):.2f}deg  ({delta_abs:.4f} rad)")
            print(f"  >>> STEPS_PER_RAD = {spr:.0f}")
            print(f"  Durchschnitt ({len(rad_results)} Messung(en)): {avg:.0f}")

        elif key == ord('+'):
            DRIVE_STEPS += 50
            print(f"DRIVE_STEPS = {DRIVE_STEPS}")

        elif key == ord('-'):
            DRIVE_STEPS = max(50, DRIVE_STEPS - 50)
            print(f"DRIVE_STEPS = {DRIVE_STEPS}")

    # ── Zusammenfassung ────────────────────────────────────────────
    print()
    print("=" * 55)
    print("Kalibrierungsergebnis:")
    if cell_results:
        avg = sum(cell_results) / len(cell_results)
        vals = "  ".join(f"{v:.0f}" for v in cell_results)
        print(f"  STEPS_PER_CELL = {avg:.0f}   (Messungen: {vals})")
    else:
        print("  STEPS_PER_CELL = nicht gemessen")

    if rad_results:
        avg = sum(rad_results) / len(rad_results)
        vals = "  ".join(f"{v:.0f}" for v in rad_results)
        print(f"  STEPS_PER_RAD  = {avg:.0f}   (Messungen: {vals})")
    else:
        print("  STEPS_PER_RAD  = nicht gemessen")

    print()
    print("→  Werte in control/kinematics.py bei STEPS_PER_CELL / STEPS_PER_RAD eintragen!")
    print("=" * 55)

    agv.abort()
    agv.disable()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
