import numpy as np
import matplotlib.pyplot as plt


def calculate_centroid(coordinates):
    return np.mean(coordinates, axis=0)


def map_to_grid(coordinates, grid_size):
    centroid = calculate_centroid(coordinates)
    distances_to_centroid = np.linalg.norm(coordinates - centroid, axis=1)
    sorted_indices = np.argsort(distances_to_centroid)
    sorted_points = coordinates[sorted_indices]

    grid_x = np.linspace(0, 200, grid_size + 1)
    grid_y = np.linspace(0, 200, grid_size + 1)

    grid = np.full((grid_size, grid_size), -1)
    mapped_coords = []

    for idx, point in zip(sorted_indices, sorted_points):
        x_idx = np.searchsorted(grid_x, point[0]) - 1
        y_idx = np.searchsorted(grid_y, point[1]) - 1

        x_idx = min(x_idx, grid_size - 1)
        y_idx = min(y_idx, grid_size - 1)

        if grid[x_idx, y_idx] == -1:
            grid[x_idx, y_idx] = idx
            mapped_coords.append(
                (grid_x[x_idx] + grid_x[1] / 2, grid_y[y_idx] + grid_y[1] / 2, idx)
            )
        else:
            for i in range(grid_size):
                for j in range(grid_size):
                    if grid[(x_idx + i) % grid_size, (y_idx + j) % grid_size] == -1:
                        grid[(x_idx + i) % grid_size, (y_idx + j) % grid_size] = idx
                        mapped_coords.append(
                            (
                                grid_x[(x_idx + i) % grid_size] + grid_x[1] / 2,
                                grid_y[(y_idx + j) % grid_size] + grid_y[1] / 2,
                                idx,
                            )
                        )
                        break
                else:
                    continue
                break

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
