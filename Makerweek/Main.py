import cv2
import math
import time
import threading
import queue as _qmod

import numpy as np

from vision.camera import Camera
from vision.markers import detect_corner_markers, detect_markers, last_corners
from vision.rectify import compute_warp_matrix, PADDING
from vision.walls import detect_walls
from vision.pose import compute_pose, world_to_grid, grid_to_world
from planning.grid import mask_to_grid
from planning.astar import astar
from control.kinematics import (execute_next_segment, flip_turn_sign,
                                 reset_calibration, cancel_calibration)
from control.agv_api import AGV
from utils.visualization import show_debug

# ============================================================
# KONFIGURATION
# ============================================================
AGV_IP            = "172.17.1.73"
CAMERA_URL        = "http://10.250.150.224:8081/"
MAX_VELOCITY_PERC = 14    # Fahrgeschwindigkeit (min ~13% damit Motoren anlaufen)
START_VEL_PERC    = 14    # Startgeschwindigkeit (erstes Segment)
ACCEL_PER_STEP    = 0     # % Zuwachs pro Segment (0 = konstant)
WALL_DANGER_PERC  = 13    # Wand direkt voraus

# Schritt-Modus (Testrun): 't' drücken → AGV fährt Segment für Segment,
# wartet nach jedem Segment auf Enter. Ideal für Kalibrierung.
STEP_MODE_VEL     = 13    # Geschwindigkeit im Schritt-Modus
GOAL_REACH_CELLS  = 1.0   # Toleranz "Ziel erreicht" in Zellen (kleiner = fährt weiter
                          # über die Ziellinie; Ziel liegt 2 Zellen hinter der Linie)
PATH_FAIL_LIMIT   = 60    # Frames ohne Pfad → Stopp (~2 s)
WALL_AHEAD_CELLS  = 2     # Zellen voraus auf Wandprüfung
BOUNDARY_CLEAR    = 20    # Pixel Feldrand freistellen

# Einfahrt (AGV hinter Startlinie)
CREEP_STEPS       = 80    # Steps vorwärts pro Kriech-Schritt (~40% Gitterzelle)
CREEP_FRAME_LIMIT = 300   # Sicherheitsstopp nach ~10 s (bei 30 fps)

# Blind-Fahrt (wenn Marker im Fahrbetrieb verloren)
BLIND_STEPS        = 20   # Steps geradeaus pro blindem Schritt
BLIND_INTERVAL     = 10   # Frames zwischen blinden Schritten
BLIND_SAFETY_LIMIT = 20   # Frames bis Sicherheitsstopp (~3 s)
# ============================================================

# ── Konsoleneingabe im Hintergrund ───────────────────────────────────────────
_cmd_q = _qmod.SimpleQueue()

def _stdin_reader():
    while True:
        try:
            _cmd_q.put(input().strip().lower())
        except EOFError:
            break

threading.Thread(target=_stdin_reader, daemon=True).start()


# ── Hilfsfunktionen ──────────────────────────────────────────────────────────

def compute_preferred_goal(warp_matrix, W=800, H=800):
    """Ziel = 2 Zellen HINTER der unteren Eckmarker-Linie (bl–br)."""
    names = ["tl", "tr", "br", "bl"]
    pts = np.float32([[last_corners[n][0], last_corners[n][1]] for n in names])
    pts_td = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), warp_matrix).reshape(-1, 2)
    c = {n: pts_td[i] for i, n in enumerate(names)}
    goal_pt = (c["bl"] + c["br"]) / 2 + np.array([0.0, 40.0])
    gx, gy = world_to_grid(float(goal_pt[0]), float(goal_pt[1]))
    print(f"  Ziel: Gitterzelle ({gx},{gy})")
    return (gx, gy)


def goal_reached(pose, goal_grid):
    gpx, gpy = grid_to_world(goal_grid[0], goal_grid[1])
    return math.hypot(pose.x - gpx, pose.y - gpy) < GOAL_REACH_CELLS * (800 / 40)


def find_reachable_goal(grid, preferred_goal):
    gx, gy = preferred_goal
    for radius in range(8):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < 40 and 0 <= ny < 40 and grid[ny][nx] == 0:
                    return (nx, ny)
    return None


def wall_directly_ahead(grid, pose):
    if pose is None:
        return False
    cell_px = 800.0 / 40
    for d in range(1, WALL_AHEAD_CELLS + 1):
        px = pose.x + math.cos(pose.theta) * d * cell_px
        py = pose.y + math.sin(pose.theta) * d * cell_px
        gx, gy = world_to_grid(px, py)
        if 0 <= gx < 40 and 0 <= gy < 40 and grid[gy][gx] == 1:
            return True
    return False


# ── Hauptprogramm ────────────────────────────────────────────────────────────

def main():
    cam = Camera(CAMERA_URL)
    agv = AGV(AGV_IP)

    warp_matrix        = None
    last_pose          = None
    preferred_goal     = None
    state              = "idle"
    goal               = None
    grid               = None
    path_fail_frames   = 0
    segments_driven    = 0
    moving_cache       = False
    moving_check_t     = 0.0
    pose_stale_frames  = 0
    cached_path        = None
    creep_frames       = 0
    grid_tick          = 0
    running            = True
    last_other_agvs    = []
    dist_map           = None
    last_pivot         = None   # Lookahead-Pivot für Visualisierung
    step_mode          = False  # 't': Segment für Segment, Enter = nächster Schritt
    step_waiting       = False  # wartet auf Enter im Schritt-Modus
    last_frame_ts      = -1.0   # Zeitstempel des letzten verarbeiteten Kamerabilds
    topdown            = None   # letztes Top-Down-Bild (für Anzeige zwischen Bildern)
    stopped_ts         = 0.0    # Zeitpunkt an dem das AGV zuletzt stehen blieb

    print("=" * 50)
    print("AGV Steuerung")
    print("=" * 50)
    print("Warte auf Eckmarker (IDs 1-4)...")
    print()
    print("  TERMINAL-EINGABE (Enter drücken):")
    print("    s  → Start (Normal)")
    print("    t  → Testrun (Schritt für Schritt, Enter = nächstes Segment)")
    print("    i  → Drehrichtung umkehren (falls AGV falsch herum dreht)")
    print("    c  → Kalibrierung zurücksetzen (Standardwerte)")
    print("    q  → Beenden")
    print("  (STEPS_PER_RAD / STEPS_PER_CELL / Drehrichtung kalibrieren sich")
    print("   beim Fahren automatisch und werden in calibration.json gespeichert)")
    print()

    while running:
        # ── Konsolen-Befehle ──────────────────────────────────────────
        cmd_start = False
        while not _cmd_q.empty():
            c = _cmd_q.get_nowait()
            if c == 's':
                cmd_start = True
                step_mode = False
                print("[Terminal] 's' – Normalstart")
            elif c == 't':
                cmd_start = True
                step_mode = True
                step_waiting = False
                print(f"[Terminal] 't' – Testrun ({STEP_MODE_VEL}%, Segment für Segment)")
            elif c == 'i':
                flip_turn_sign()
            elif c == 'c':
                reset_calibration()
            elif c == '':
                # leere Eingabe (Enter allein) → nächsten Schritt freigeben
                if step_mode and step_waiting:
                    step_waiting = False
                    print("[STEP] Weiter →")
            elif c == 'q':
                running = False
                print("[Terminal] 'q' empfangen – beende...")

        frame, frame_ts = cam.get_frame()
        if frame is None:
            cv2.waitKey(1)
            continue
        # Kamera läuft nur ~1 FPS → erkennen, ob ein NEUES Bild vorliegt.
        new_frame     = (frame_ts != last_frame_ts)
        last_frame_ts = frame_ts

        # ── Eckmarker & Warp (einmal cachen) ──────────────────────────
        if warp_matrix is None:
            corners = detect_corner_markers(frame)
            if corners is None:
                show_debug(cv2.resize(frame, (800, 800)), None, None, None, "no_marker")
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                continue
            warp_matrix    = compute_warp_matrix(corners)
            preferred_goal = compute_preferred_goal(warp_matrix)
            print("Spielfeld erkannt!")
            print(">>> Drücke 's' + Enter im Terminal zum Starten <<<")

        # ── Bildverarbeitung NUR bei neuem Kamerabild ─────────────────
        # Sonst würde dasselbe Standbild zig-mal ausgewertet: pose_stale_frames
        # zählt pro Iteration hoch (Fehlalarm-Stopp), 3× ArUco pro Iteration.
        if new_frame or topdown is None:
            topdown = cv2.warpPerspective(frame, warp_matrix, (800, 800))

            # ── Wände & Grid (nur jedes 3. Kamerabild neu berechnen) ──
            grid_tick += 1
            if grid_tick % 3 == 0 or grid is None:
                wall_mask = detect_walls(topdown)
                W = H = 800
                for px, py in [(PADDING, PADDING), (W - PADDING, PADDING),
                               (W - PADDING, H - PADDING), (PADDING, H - PADDING)]:
                    cv2.circle(wall_mask, (px, py), 40, 0, -1)
                B = BOUNDARY_CLEAR
                wall_mask[:PADDING + B, :]     = 0
                wall_mask[H - PADDING - B:, :] = 0
                wall_mask[:, :PADDING + B]     = 0
                wall_mask[:, W - PADDING - B:] = 0
                # Eigenes AGV aus Wandmaske entfernen (Ellipse entlang Fahrtrichtung)
                if last_pose is not None:
                    theta = last_pose.theta
                    ec_x  = int(last_pose.x - math.cos(theta) * 20)
                    ec_y  = int(last_pose.y - math.sin(theta) * 20)
                    cv2.ellipse(wall_mask, (ec_x, ec_y), (48, 24),
                                math.degrees(theta), 0, 360, 0, -1)
                # Fremde AGVs komplett aus Wandmaske löschen (100px = ~5 Zellen)
                for _, (fx, fy) in last_other_agvs:
                    cv2.circle(wall_mask, (int(fx), int(fy)), 100, 0, -1)
                grid     = mask_to_grid(wall_mask)
                free_img = ((grid == 0) * 255).astype(np.uint8)
                dist_map = cv2.distanceTransform(free_img, cv2.DIST_L2, 3)

            # ── AGV-Pose ──────────────────────────────────────────────
            agv_marker, other_agvs = detect_markers(frame, warp_matrix, topdown,
                                                    stale_frames=pose_stale_frames)
            last_other_agvs = other_agvs
            pose = compute_pose(agv_marker)
            if pose is not None:
                last_pose         = pose
                pose_stale_frames = 0
            else:
                pose_stale_frames += 1
                pose = last_pose
        else:
            # Zwischen zwei Kamerabildern: letzte bekannte Pose behalten (kein Raten)
            pose = last_pose

        wall_close = wall_directly_ahead(grid, pose)

        # ── State Machine ─────────────────────────────────────────────
        if state == "starting":
            if new_frame:
                creep_frames += 1
            if creep_frames >= CREEP_FRAME_LIMIT:
                agv.abort()
                agv.disable()
                state = "idle"
                print("Timeout – kein Marker gefunden. Zurück zu idle.")

            elif pose_stale_frames == 0:
                # Marker frisch erkannt → sofort stoppen und Pfad berechnen
                agv.abort()
                agv.wait_for_stop(timeout=2)   # erst vollständig stoppen
                start_cell  = world_to_grid(pose.x, pose.y)
                goal        = find_reachable_goal(grid, preferred_goal)
                if goal is not None:
                    cached_path = astar(grid, start_cell, goal, dist_map)
                moving_cache = False
                stopped_ts   = time.time()
                state        = "driving"
                print(f"Marker erkannt @ ({pose.x:.0f},{pose.y:.0f}) – starte Pfad!")

            else:
                # Kein Marker: kleiner Schritt vor – aber erst wenn AGV steht UND
                # ein frisches Kamerabild vorliegt (bei ~1 FPS sonst blind-Dauerfahrt)
                now = time.time()
                if now - moving_check_t >= 0.25:
                    prev_moving    = moving_cache
                    moving_cache   = agv.is_moving()
                    moving_check_t = now
                    if prev_moving and not moving_cache:
                        stopped_ts = now
                if not moving_cache and frame_ts > stopped_ts:
                    agv.set_max_acceleration(20)
                    agv.set_max_velocity(15)
                    agv.drive(CREEP_STEPS, CREEP_STEPS)
                    moving_cache   = True
                    moving_check_t = time.time()

        elif state == "driving":

            if pose_stale_frames >= BLIND_SAFETY_LIMIT:
                # Zu lange blind → Sicherheitsstopp
                state = "idle"
                path_fail_frames  = 0
                segments_driven   = 0
                moving_cache      = False
                pose_stale_frames = 0
                cached_path       = None
                print("Marker zu lange unsichtbar – Sicherheitsstopp.")
                agv.abort()
                agv.disable()

            elif pose_stale_frames > 0:
                # Kein Marker: langsam geradeaus tappen (OHNE drehen)
                cancel_calibration()   # blinde Taps verfälschen sonst die Messung
                now = time.time()
                if now - moving_check_t >= 0.25:
                    moving_cache   = agv.is_moving()
                    moving_check_t = now
                if not moving_cache and pose_stale_frames % BLIND_INTERVAL == 0:
                    agv.set_max_acceleration(20)
                    agv.set_max_velocity(15)
                    agv.drive(BLIND_STEPS, BLIND_STEPS)
                    moving_cache   = True
                    moving_check_t = time.time()

            # Ziel erreicht?
            elif goal is not None and goal_reached(pose, goal):
                state = "finished"
                path_fail_frames = 0
                segments_driven  = 0
                moving_cache     = False
                cached_path      = None
                print("*** ZIEL ERREICHT! ***")
                agv.abort()
                agv.disable()

            else:
                # is_moving() max alle 250 ms abfragen
                now = time.time()
                if now - moving_check_t >= 0.25:
                    prev_moving    = moving_cache
                    moving_cache   = agv.is_moving()
                    moving_check_t = now
                    if prev_moving and not moving_cache:
                        stopped_ts = now   # AGV gerade stehengeblieben

                if step_mode and step_waiting:
                    pass  # warte auf Enter im Terminal

                elif moving_cache:
                    pass  # fährt noch – nicht stören

                elif frame_ts <= stopped_ts:
                    pass  # Kamera ~1 FPS: auf frisches Bild NACH dem Stopp warten,
                          # bevor der nächste Zug geplant wird (kein blindes Durchfahren)

                else:
                    # AGV steht + frisches Kamerabild liegt vor → nächsten Zug planen
                    # Wegpunkte überspringen die das AGV bereits passiert hat
                    if pose is not None and cached_path and len(cached_path) > 1:
                        while len(cached_path) > 1:
                            wp = grid_to_world(cached_path[1][0], cached_path[1][1])
                            if math.hypot(pose.x - wp[0], pose.y - wp[1]) < (800/40) * 0.9:
                                cached_path = cached_path[1:]
                            else:
                                break

                    # Nur replanen wenn Pfad erschöpft (< 2 Wegpunkte übrig)
                    if cached_path is None or len(cached_path) < 2:
                        if pose is not None and preferred_goal is not None:
                            start_cell = world_to_grid(pose.x, pose.y)
                            goal       = find_reachable_goal(grid, preferred_goal)
                            if goal is not None:
                                cached_path = astar(grid, start_cell, goal, dist_map)

                    path = cached_path
                    if path is None or len(path) < 2:
                        path_fail_frames += 1
                        if path_fail_frames >= PATH_FAIL_LIMIT:
                            state = "idle"
                            path_fail_frames = 0
                            segments_driven  = 0
                            moving_cache     = False
                            cached_path      = None
                            print("Kein Pfad gefunden – gestoppt.")
                            agv.abort()
                            agv.disable()
                    else:
                        path_fail_frames = 0
                        if step_mode:
                            vel = STEP_MODE_VEL
                        else:
                            accel_vel = min(MAX_VELOCITY_PERC,
                                            START_VEL_PERC + segments_driven * ACCEL_PER_STEP)
                            vel = WALL_DANGER_PERC if (wall_close and segments_driven >= 2) else accel_vel
                        count, last_pivot, _ = execute_next_segment(agv, pose, path, grid=grid, max_vel=vel)
                        cached_path = cached_path[count:] if len(cached_path) > count else None
                        segments_driven += 1
                        moving_cache    = True
                        moving_check_t  = time.time()
                        if step_mode:
                            remaining = len(cached_path) if cached_path else 0
                            print(f"[STEP] Segment {segments_driven} fertig  "
                                  f"Pfad noch {remaining} Zellen  "
                                  f"→ Enter für nächsten Schritt (oder 'q')")
                            step_waiting = True

        # ── Status bestimmen ──────────────────────────────────────────
        path = cached_path
        if state == "finished":
            status = "goal_reached"
        elif state == "starting":
            status = "starting"
        elif state == "driving":
            if pose_stale_frames > 0:
                status = "pose_lost"
            elif wall_close:
                status = "danger"
            else:
                status = "driving"
        elif pose is None:
            status = "no_marker"
        elif path is not None and len(path) >= 2:
            status = "ready"
        else:
            status = "no_path"

        # ── Display ───────────────────────────────────────────────────
        show_debug(topdown, grid, pose, path, status, goal,
                   other_agvs=other_agvs, pivot=last_pivot)

        # ── Tastatur (CV2-Fenster) ────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            running = False
        if key == ord('s'):
            cmd_start = True

        # ── Start-Befehl ──────────────────────────────────────────────
        if cmd_start and state in ("idle", "finished"):
            path_fail_frames = 0
            segments_driven  = 0
            cached_path      = None
            moving_cache     = False
            creep_frames     = 0
            cancel_calibration()   # keine veraltete Messung vom letzten Lauf
            agv.enable()
            if pose is not None and pose_stale_frames == 0 and grid is not None:
                # AGV schon sichtbar → sofort A* berechnen und direkt fahren
                time.sleep(0.1)
                start_cell  = world_to_grid(pose.x, pose.y)
                goal        = find_reachable_goal(grid, preferred_goal)
                if goal is not None:
                    cached_path = astar(grid, start_cell, goal, dist_map)
                last_pivot = None
                state = "driving"
                print(f"Marker sichtbar @ ({pose.x:.0f},{pose.y:.0f}) – sofort Pfad!")
            else:
                time.sleep(0.3)
                state = "starting"
                print("STARTE – kriecht vor bis AGV-Marker erkannt...")

    agv.abort()
    agv.disable()
    cv2.destroyAllWindows()
    print("Programm beendet.")


if __name__ == "__main__":
    main()
