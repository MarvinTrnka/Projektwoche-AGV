import math

STEPS_PER_CM = 20
STEPS_PER_RAD = 100

def turn_to_angle(agv, pose, waypoint):
    dx = waypoint[0] - pose.x
    dy = waypoint[1] - pose.y
    target_angle = math.atan2(dy, dx)
    error = target_angle - pose.theta

    steps = int(error * STEPS_PER_RAD)
    agv.drive(-steps, steps)

def drive_distance(agv, pose, waypoint):
    dx = waypoint[0] - pose.x
    dy = waypoint[1] - pose.y
    dist = math.sqrt(dx*dx + dy*dy)

    steps = int(dist * STEPS_PER_CM)
    agv.drive(steps, steps)

def path_to_commands(path):
    cmds = []
    for (x1,y1),(x2,y2) in zip(path, path[1:]):
        dx = x2 - x1
        dy = y2 - y1

        if dx == 1: cmds.append("RIGHT")
        if dx == -1: cmds.append("LEFT")
        if dy == 1: cmds.append("DOWN")
        if dy == -1: cmds.append("UP")

    return cmds
