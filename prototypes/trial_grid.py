import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import matplotlib.cm as cm


def calculate_centroid(coordinates):
    return np.mean(coordinates, axis=0)


def map_to_grid(coordinates, grid_size):
    num_points = len(coordinates)

    # Initialize a grid with -1, this grid is only used for visualization in this context
    grid = np.full((grid_size, grid_size), -1)

    points = np.array(coordinates)

    # Calculate the center of the provided coordinates
    grid_center = calculate_centroid(points)

    # Define Grid Points
    grid_coords = np.linspace(0, 200, grid_size)
    grid_points = np.array([(x, y) for x in grid_coords for y in grid_coords])

    # Calculate Distances to Grid and Sort by Priority
    distances_to_center = np.linalg.norm(points - grid_center, axis=1)
    sorted_indices = np.argsort(distances_to_center)  # prioritize closer to center
    sorted_points = points[sorted_indices]

    # Generate colors for sorted points
    colors = cm.plasma(np.linspace(0, 1, num_points))
    sorted_colors = colors[sorted_indices]

    # Track allocated grid points
    allocated_grid = set()

    mapped_coords = []  # To store the final mapped positions

    for i, point in enumerate(sorted_points):  # Iterate over sorted points
        dists = distance.cdist([point], grid_points, metric="euclidean").flatten()
        sorted_grid_indices = np.argsort(dists)

        # Find the closest unallocated grid point
        for idx in sorted_grid_indices:
            grid_point = tuple(grid_points[idx])
            if grid_point not in allocated_grid:
                grid_flat_index = idx
                grid_x, grid_y = divmod(grid_flat_index, grid_size)
                grid[grid_x, grid_y] = i  # Record the index for visualization
                mapped_coords.append(
                    (grid_x * (200 / (grid_size - 1)), grid_y * (200 / (grid_size - 1)))
                )
                allocated_grid.add(grid_point)
                break

    return grid, sorted_colors, mapped_coords


# Example Usage
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
mapped_grid, colors, mapped_coords = map_to_grid(coordinates, grid_size)

# Plotting the result
fig, ax = plt.subplots(figsize=(10, 8))

# Plot original coordinates with colors
for i, point in enumerate(coordinates):
    ax.scatter(
        point[0], point[1], color=colors[i], s=100, label=f"Point {i}" if i < 1 else ""
    )

# Plot grid points with same colors
for i, (grid_x, grid_y) in enumerate(mapped_coords):
    ax.scatter(
        grid_x,
        grid_y,
        color=colors[i],
        s=100,
        marker="x",
        label=f"Mapped {i}" if i < 1 else "",
    )

# Plot grid lines without the scatter plot grid lines
grid_lines = np.linspace(0, 200, grid_size)
for line in grid_lines:
    ax.axhline(line, color="gray", linestyle="--", linewidth=0.5)
    ax.axvline(line, color="gray", linestyle="--", linewidth=0.5)

ax.set_title("Mapped 2D Coordinates with Grid")
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.legend(loc="upper right", fontsize="small")
ax.set_xlim(0, 200)
ax.set_ylim(0, 200)
ax.set_aspect("equal")
ax.grid(False)  # Remove default grid lines

plt.show()
