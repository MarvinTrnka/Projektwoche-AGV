import math
from vision.pose import grid_to_world

# ============================================================
# Kalibrierungskonstanten — durch Testfahrten anpassen!
# ============================================================
STEPS_PER_CELL     = 200    # Steps für eine Gitterzelle vorwärts (~5 cm)
STEPS_PER_RAD      = 83     # Steps pro Radian
TURN_VELOCITY_PERC = 15     # Geschwindigkeit beim Drehen (langsamer = präziser)
MIN_TURN_RAD       = math.radians(5)   # Drehungen < 5° überspringen
CELL_PX            = 800 / 40          # Pixel pro Gitterzelle = 20 px
MAX_SEGMENT_CELLS  = 2      # Max Zellen pro Segment → Kurskorrektur alle 2 Zellen
LATERAL_CORR_DEG   = 6.0   # Grad Kurskorrektur pro Zelle Wandabstand-Ungleichgewicht
# ============================================================


def _normalize_angle(a):
    return ((a + math.pi) % (2 * math.pi)) - math.pi


def _lateral_correction(pose, grid):
    """Berechnet Kurskorrektur basierend auf Wandabstand links/rechts.

    Positiv = nach rechts steuern (weg von linker Wand).
    Negativ = nach links steuern (weg von rechter Wand).
    """
    left_angle  = pose.theta - math.pi / 2
    right_angle = pose.theta + math.pi / 2
    left_clear  = 3
    right_clear = 3

    for d in range(1, 4):
        lx = pose.x + math.cos(left_angle) * d * CELL_PX
        ly = pose.y + math.sin(left_angle) * d * CELL_PX
        gx = max(0, min(39, int(lx / CELL_PX)))
        gy = max(0, min(39, int(ly / CELL_PX)))
        if grid[gy][gx] == 1:
            left_clear = d - 1
            break

    for d in range(1, 4):
        rx = pose.x + math.cos(right_angle) * d * CELL_PX
        ry = pose.y + math.sin(right_angle) * d * CELL_PX
        gx = max(0, min(39, int(rx / CELL_PX)))
        gy = max(0, min(39, int(ry / CELL_PX)))
        if grid[gy][gx] == 1:
            right_clear = d - 1
            break

    # diff > 0 → mehr Platz rechts → näher an linker Wand → nach rechts korrigieren
    diff = right_clear - left_clear
    return math.radians(diff * LATERAL_CORR_DEG)


def turn_to_angle(agv, pose, waypoint_pixel, extra_angle=0.0):
    """Dreht auf der Stelle zum Wegpunkt ± optionaler Kurskorrektur."""
    dx = waypoint_pixel[0] - pose.x
    dy = waypoint_pixel[1] - pose.y
    target_angle = math.atan2(dy, dx) + extra_angle
    error = _normalize_angle(target_angle - pose.theta)
    if abs(error) < MIN_TURN_RAD:
        return
    steps = int(error * STEPS_PER_RAD)
    if steps != 0:
        # drive(steps, -steps):
        #   steps > 0 → Rechtsdrehung → theta steigt ✓
        #   steps < 0 → Linksdrehung  → theta sinkt ✓
        agv.drive(steps, -steps)
        agv.wait_for_stop()


def execute_next_segment(agv, pose, path, grid=None, max_vel=60):
    """Dreht einmal (blockierend) und fährt dann max. MAX_SEGMENT_CELLS Zellen.

    grid: optionales 40×40-Gitter für laterale Wandkorrektur.
    """
    dx = path[1][0] - path[0][0]
    dy = path[1][1] - path[0][1]

    # Aufeinanderfolgende gleiche Richtung bündeln (max MAX_SEGMENT_CELLS)
    count = 1
    while count < min(len(path) - 1, MAX_SEGMENT_CELLS):
        ndx = path[count + 1][0] - path[count][0]
        ndy = path[count + 1][1] - path[count][1]
        if (ndx, ndy) != (dx, dy):
            break
        count += 1

    target_pixel = grid_to_world(path[count][0], path[count][1])

    # Laterale Wandkorrektur: kleiner Zusatzwinkel wenn nah an Seitenwand
    correction = _lateral_correction(pose, grid) if grid is not None else 0.0

    # Drehen mit Kurskorrektur
    agv.set_max_acceleration(15)
    agv.set_max_velocity(TURN_VELOCITY_PERC)
    turn_to_angle(agv, pose, target_pixel, extra_angle=correction)

    # Distanz von aktueller Pose zur Zielzelle
    dist_px    = math.hypot(target_pixel[0] - pose.x, target_pixel[1] - pose.y)
    drive_steps = max(1, int(dist_px * STEPS_PER_CELL / CELL_PX))

    agv.set_max_acceleration(15)
    agv.set_max_velocity(max_vel)
    agv.drive(drive_steps, drive_steps)   # nicht blockierend
    return count
