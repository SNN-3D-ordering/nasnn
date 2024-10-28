from audioop import reverse
import numpy as np
import matplotlib.pyplot as plt


def make_grid_from_coordinates(coordinates, pad_value=-1):
    """
    Take a set of continuous values, generate a 2d grid off them.
    """
    # Calculate grid params, init grid
    num_coords = len(coordinates)
    grid_side_length = int(np.ceil(np.sqrt(num_coords)))
    grid = np.full((grid_side_length, grid_side_length), pad_value, dtype=int)

    # Store idx and coordinates
    coordinates_list = [[idx, [x, y]] for idx, (x, y) in enumerate(coordinates)]

    # Sort the coordinates list according to y value
    coordinates_list_sorted_by_y = sorted(coordinates_list, key=lambda item: item[1][1])

    # Take side_length sorted_y_indices at a time, sort them by their x_value, and take that as one row
    rows = []
    for i in range(0, num_coords, grid_side_length):
        chunk = coordinates_list_sorted_by_y[i : i + grid_side_length]
        chunk_sorted = sorted(chunk, key=lambda x: x[1][0])
        rows.append([idx for idx, _ in chunk_sorted])

    # Write to grid
    for row_idx, sublist in enumerate(reversed(rows)):
        grid[row_idx, : len(sublist)] = sublist

    return grid


# test
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
    ]
)

mapped_coords = make_grid_from_coordinates(coordinates)
print(mapped_coords)

# Plot the previous and new coordinates
plt.figure(figsize=(10, 5))

# Original coordinates plot
plt.subplot(121)
plt.scatter(coordinates[:, 0], coordinates[:, 1])
plt.title("Original coordinates")
for i, (x, y) in enumerate(coordinates):
    plt.annotate(str(i), (x, y), textcoords="offset points", xytext=(0, 5), ha="center")


plt.tight_layout()
plt.show()
