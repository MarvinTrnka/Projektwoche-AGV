import numpy as np

def mask_to_grid(mask, grid_size=40):
    H, W = mask.shape
    cell_h = H // grid_size
    cell_w = W // grid_size

    grid = np.zeros((grid_size, grid_size), dtype=np.uint8)

    for gy in range(grid_size):
        for gx in range(grid_size):
            y1 = gy * cell_h
            y2 = y1 + cell_h
            x1 = gx * cell_w
            x2 = x1 + cell_w

            cell = mask[y1:y2, x1:x2]

            if cell.mean() > 20:
                grid[gy][gx] = 1

    return grid
