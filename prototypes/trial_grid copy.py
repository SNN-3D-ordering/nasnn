import numpy as np
import matplotlib.pyplot as plt


def map_to_grid(coordinates, grid_size):
    """
    Map a set of 2D coordinates onto a grid of specified size. The function
    attempts to place each point in a grid cell, starting from the cell closest
    to the centroid and finding the nearest free cell if needed.

    Parameters:
        coordinates (array-like): An array of 2D points, each representing a coordinate (x, y).
        grid_size (int): The size of the grid (grid_size x grid_size).

    Returns:
        grid (np.array): A 2D array of the grid with indices of mapped coordinates.
        mapped_coords (list of tuples): A list of tuples containing the grid center (x, y)
                                        and the original index of the coordinate.
    """
    # Calculate the centroid of the input coordinates (mean along each dimension)
    centroid = np.mean(coordinates, axis=0)

    # Calculate the Euclidean distance of each point to the centroid
    distances_to_centroid = np.linalg.norm(coordinates - centroid, axis=1)
    # Get indices of coordinates sorted by their distance to the centroid
    sorted_indices = np.argsort(distances_to_centroid)

    # Determine the min and max values along x and y axes to gauge range of coordinates
    min_x, min_y = np.min(coordinates, axis=0)
    max_x, max_y = np.max(coordinates, axis=0)

    # Create grid ranges dynamically based on min and max values of coordinates
    grid_x = np.linspace(min_x, max_x, grid_size + 1)
    grid_y = np.linspace(min_y, max_y, grid_size + 1)

    # Initialize a grid array with -1 indicating empty cells
    grid = np.full((grid_size, grid_size), -1)

    # List to store the map of coordinates to grid locations
    mapped_coords = []

    # Calculate the halfway point of the grid size
    half_grid = grid_size // 2

    # Loop through the indices of points sorted by proximity to the centroid
    for idx in sorted_indices:
        point = coordinates[idx]

        # Compute initial grid cell coordinates based on the transformed position about the centroid
        rel_x = min(
            max(
                int(
                    (point[0] - centroid[0]) / ((max_x - min_x) / grid_size) + half_grid
                ),
                0,
            ),
            grid_size - 1,
        )
        rel_y = min(
            max(
                int(
                    (point[1] - centroid[1]) / ((max_y - min_y) / grid_size) + half_grid
                ),
                0,
            ),
            grid_size - 1,
        )

        # Attempt to place it at calculated cell, else find nearest free cell
        placed = False

        # Loop over the range of possible offsets (distance from the initial target cell)
        for offset in range(grid_size):
            # Iterate over horizontal offsets (-offset to +offset)
            for dx in range(-offset, offset + 1):
                # Iterate over vertical offsets (-offset to +offset)
                for dy in range(-offset, offset + 1):
                    row_idx, col_idx = (rel_y + dy) % grid_size, (
                        rel_x + dx
                    ) % grid_size

                    # If the cell is empty, place the point there
                    if grid[row_idx, col_idx] == -1:
                        grid[row_idx, col_idx] = idx

                        # Calculate the center of the grid cell for final mapped coordinates
                        mapped_coords.append(
                            (
                                grid_x[col_idx] + grid_x[1] / 2,
                                grid_y[row_idx] + grid_y[1] / 2,
                                idx,
                            )
                        )
                        # Mark as placed and break out of loops
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
