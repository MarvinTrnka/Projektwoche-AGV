import heapq

def astar(grid, start, goal):
    W = len(grid[0])
    H = len(grid)

    def neighbors(x, y):
        for nx, ny in [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]:
            if 0 <= nx < W and 0 <= ny < H:
                if grid[ny][nx] == 0:
                    yield nx, ny

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for n in neighbors(*current):
            new_cost = g[current] + 1
            if n not in g or new_cost < g[n]:
                g[n] = new_cost
                f = new_cost + abs(n[0]-goal[0]) + abs(n[1]-goal[1])
                heapq.heappush(open_set, (f, n))
                came_from[n] = current

    return None
