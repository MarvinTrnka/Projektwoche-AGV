import heapq

WALL_BIAS = 3.0  # höher = stärker zur Mitte der Korridore hin

def astar(grid, start, goal, dist_map=None):
    W = len(grid[0])
    H = len(grid)

    def neighbors(x, y):
        for nx, ny in [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]:
            if 0 <= nx < W and 0 <= ny < H:
                if grid[ny][nx] == 0:
                    yield nx, ny

    open_set = []
    heapq.heappush(open_set, (0.0, start))
    came_from = {}
    g      = {start: 0.0}
    closed = set()

    while open_set:
        _, current = heapq.heappop(open_set)
        if current in closed:
            continue
        closed.add(current)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        for n in neighbors(*current):
            if n in closed:
                continue
            dist  = float(dist_map[n[1]][n[0]]) if dist_map is not None else 1.0
            cost  = 1.0 + WALL_BIAS / (dist + 0.5)
            new_g = g[current] + cost
            if n not in g or new_g < g[n]:
                g[n] = new_g
                f    = new_g + abs(n[0]-goal[0]) + abs(n[1]-goal[1])
                heapq.heappush(open_set, (f, n))
                came_from[n] = current

    return None
