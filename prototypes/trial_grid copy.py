import numpy as np
import matplotlib.pyplot as plt


def map_to_grid(coordinates, grid_size):
    centroid = np.mean(coordinates, axis=0)

    # Determine the range of coordinates
    min_x, min_y = np.min(coordinates, axis=0)
    max_x, max_y = np.max(coordinates, axis=0)

    # Calculate the scaling factors based on the coordinate ranges
    range_x = max_x - min_x
    range_y = max_y - min_y

    # Calculate distances and argsort
    distances_to_centroid = np.linalg.norm(coordinates - centroid, axis=1)
    sorted_indices = np.argsort(distances_to_centroid)

    # Initialize grid definition
    grid_x = np.linspace(min_x, max_x, grid_size + 1)
    grid_y = np.linspace(min_y, max_y, grid_size + 1)
    grid = np.full((grid_size, grid_size), -1)

    mapped_coords = []

    # Normalize the coordinate mapping to fit the grid dimensions and start placement
    for idx in sorted_indices:
        point = coordinates[idx]

        # Map coordinates to grid indices via normalization
        rel_x = int((point[0] - min_x) / range_x * (grid_size - 1))
        rel_y = int((point[1] - min_y) / range_y * (grid_size - 1))

        # Attempt to place it at calculated cell, else find nearest free cell
        placed = False
        for offset in range(grid_size):
            for dx in range(-offset, offset + 1):
                for dy in range(-offset, offset + 1):
                    row_idx, col_idx = (rel_y + dy) % grid_size, (
                        rel_x + dx
                    ) % grid_size
                    if grid[row_idx, col_idx] == -1:
                        grid[row_idx, col_idx] = idx
                        mapped_coords.append(
                            (
                                grid_x[col_idx] + grid_x[1] / 2,
                                grid_y[row_idx] + grid_y[1] / 2,
                                idx,
                            )
                        )
                        placed = True
                        break
                if placed:
                    break
            if placed:
                break
    print(grid)
    return grid, mapped_coords


coordinates = np.array(
    [
        [95.5, 105.4],
        [100.2, 110.1],
        [102.3, 98.7],
        [99.8, 102.5],
        [101.1, 95.8],
        [110.6, 90.3],
        [90.7, 108.9],
        [97.2, 95.5],
        [94.1, 99.6],
        [105.3, 102.9],
        [98.4, 101.8],
        [100.7, 100.5],
        [105.8, 107.2],
        [102.9, 93.4],
        [96.6, 106.3],
        [108.1, 104.7],
        [92.8, 97.1],
        [99.9, 98.4],
        [101.2, 102.7],
        [97.5, 103.8],
        [96.3, 94.2],
        [104.6, 100.1],
        [95.4, 95.7],
        [93.2, 97.4],
        [107.7, 99.5],
        [108.4, 96.2],
        [99.1, 110.8],
        [104.5, 110.6],
        [103.8, 105.5],
        [100.9, 120.7],
        [85.3, 110.2],
        [70.8, 130.1],
        [90.9, 90.1],
        [80.6, 100.3],
        [75.1, 115.2],
        [110.7, 85.4],
        [120.2, 65.5],
        [102.3, 88.9],
        [98.5, 90.6],
        [115.8, 80.2],
        [115.5, 100.1],
        [85.4, 80.8],
        [90.3, 120.5],
        [110.9, 75.3],
        [100.1, 70.7],
        [120.6, 100.9],
        [60.2, 105.8],
        [120.3, 95.3],
        [100.8, 50.4],
        [130.5, 110.2],
        [125.3, 130.7],
        [110.8, 130.4],
        [60.1, 100.5],
        [150.9, 150.6],
        [200.7, 200.3],
        [115.5, 60.7],
        [110.8, 50.9],
        [120.4, 55.2],
        [95.8, 55.3],
        [90.2, 45.1],
        [80.6, 50.2],
        [75.4, 65.8],
        [70.1, 75.3],
        [60.9, 90.2],
        [55.7, 100.4],
        [100.3, 125.5],
        [105.8, 130.2],
        [70.3, 90.9],
        [65.1, 75.6],
        [50.4, 100.8],
        [45.3, 105.7],
        [50.9, 125.1],
        [30.2, 130.5],
        [20.1, 100.9],
        [10.4, 95.7],
        [25.6, 60.9],
        [120.7, 80.6],
        [150.3, 30.9],
        [180.1, 150.2],
        [200.5, 60.8],
        [0.6, 0.7],
        [10.3, 10.5],
        [30.8, 30.9],
        [50.6, 50.5],
        [70.7, 70.3],
        [90.8, 90.1],
    ]
)


grid_size = 10
mapped_grid, mapped_coords = map_to_grid(coordinates, grid_size)

# Plotting the result
fig, ax = plt.subplots(figsize=(12, 10))

# Plot original coordinates with indices
for i, point in enumerate(coordinates):
    ax.scatter(
        point[0], point[1], s=100, color="blue", label=f"Point {i}" if i == 0 else ""
    )
    ax.text(
        point[0], point[1], f"{i}", fontsize=8, ha="center", va="center", color="white"
    )

# Plot grid points with same indices
for grid_x, grid_y, idx in mapped_coords:
    ax.scatter(
        grid_x,
        grid_y,
        s=100,
        color="red",
        marker="x",
        label=f"Mapped {idx}" if idx == 0 else "",
    )
    ax.text(
        grid_x, grid_y, f"{idx}", fontsize=8, ha="center", va="center", color="black"
    )

# Plot grid lines
for line in np.linspace(0, 200, grid_size + 1):
    ax.axhline(line, color="gray", linestyle="--", linewidth=0.5)
    ax.axvline(line, color="gray", linestyle="--", linewidth=0.5)

ax.set_title("Mapped 2D Coordinates with Grid using Indices")
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.legend(loc="upper right", fontsize="small")
ax.set_xlim(0, 200)
ax.set_ylim(0, 200)
ax.set_aspect("equal")
ax.grid(False)

plt.show()