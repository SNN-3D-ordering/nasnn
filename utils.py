import numpy as np
import torch
import matplotlib.pyplot as plt


def print_batch_accuracy(data, targets, net, batch_size, train=False):
    """
    Print the accuracy of the network on a single minibatch.

    Args:
        data (torch.Tensor): The input data to the network.
        targets (torch.Tensor): The target labels for the input data.
        net (Network): The network to evaluate.
        batch_size (int): The size of the minibatch.
        train (bool): Whether the data is from the training set or not.
    """
    output, _ = net(data.view(batch_size, -1))
    _, idx = output.sum(dim=0).max(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())

    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")


def train_printer(
    epoch,
    iter_counter,
    loss_hist,
    test_loss_hist,
    data,
    targets,
    test_data,
    test_targets,
    net,
    batch_size,
):
    print(f"Epoch {epoch}, Iteration {iter_counter}")
    print(f"Train Set Loss: {loss_hist[-1]:.2f}")
    print(f"Test Set Loss: {test_loss_hist[-1]:.2f}")
    print_batch_accuracy(data, targets, net, batch_size, train=True)
    print_batch_accuracy(test_data, test_targets, net, batch_size, train=False)
    print("\n")


def make_2d_grid_from_1d_list(list_1d):
    """
    Makes a 2d grid from a 1d list by padding with -1.

    Args:
        list_1d (list): list to convert to 2d grid

    Returns:
        np.array: 2d grid, square, padded with -1 if necessary
    """
    if not isinstance(list_1d, np.ndarray):
        list_1d = np.array(list_1d)

    next_square = get_next_square_numbers(list_1d.size)[0]
    grid_2d = np.pad(
        list_1d, (0, next_square - list_1d.size), mode="constant", constant_values=-1
    )

    return grid_2d.reshape(int(np.sqrt(next_square)), int(np.sqrt(next_square)))

    # def map_to_2d_grid_row_wise(coordinates, grid_size):
    """
    Maps 2D coordinates to a 2D grid.

    Args:
        coordinates (np.ndarray): Array of 2D coordinates.
        grid_size (tuple): Size of the grid (rows, cols).

    Returns:
        np.ndarray: 2D grid with neuron indices.
    """
    num_neurons = coordinates.shape[0]
    sorted_indices = np.argsort(
        coordinates[:, 0]
    )  # Sort by x-coordinate TODO build proper sorting
    grid = np.zeros(grid_size, dtype=int) - 1  # Initialize grid with -1 (empty)

    for idx, neuron_idx in enumerate(sorted_indices):
        row = idx // grid_size[1]
        col = idx % grid_size[1]
        grid[row, col] = neuron_idx

    return grid


def map_to_2d_grid_row_wise(coordinates, grid_size):
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
    return grid


def make_2d_grid_from_1d_list(coordinates, grid_size):
    """
    Maps neurons to a uniform grid ensuring a regular distance between each.

    Args:
        coordinates (np.ndarray): Array of 2D coordinates of neurons.
        grid_size (tuple): Size of the grid (rows, cols).

    Returns:
        np.ndarray: 2D grid with neuron indices filling the grid uniformly.
    """
    num_neurons = coordinates.shape[0]

    # Number of grid positions must be at least equal to number of neurons
    if grid_size[0] * grid_size[1] < num_neurons:
        raise ValueError("Grid is too small for the number of neurons.")

    # Calculate the center neuron according to original coordinates
    center = np.mean(coordinates, axis=0)
    distances = np.linalg.norm(coordinates - center, axis=1)
    center_index = np.argmin(distances)

    # Create a list of neuron indices
    neuron_indices = list(range(num_neurons))

    # Sort neuron indices to put center_index at first
    neuron_indices.remove(center_index)
    neuron_indices = [center_index] + neuron_indices

    # Create grid and initialize with -1 (empty)
    grid = np.zeros(grid_size, dtype=int) - 1

    # Determine grid center
    mid_row = grid_size[0] // 2
    mid_col = grid_size[1] // 2

    # Calculate step sizes based on grid size and number of neurons
    num_rows_available = min(grid_size[0], int(np.ceil(np.sqrt(num_neurons))))
    num_cols_available = min(grid_size[1], int(np.ceil(np.sqrt(num_neurons))))

    # Generate positions in the grid
    positions = [
        (i, j) for i in range(num_rows_available) for j in range(num_cols_available)
    ]

    # Shift positions to start from grid center
    start_pos_idx = len(positions) // 2
    centered_positions = [
        (r + mid_row - num_rows_available // 2, c + mid_col - num_cols_available // 2)
        for r, c in positions
    ]

    # Assign neurons to grid positions
    for idx, pos in zip(neuron_indices, centered_positions):
        r, c = pos
        if 0 <= r < grid_size[0] and 0 <= c < grid_size[1]:
            grid[r, c] = idx

    return grid


def map_to_1d_list_row_wise(coordinates):
    """
    Maps 2D coordinates to a 1D list row-wise.

    Args:
        coordinates (np.ndarray): Array of 2D coordinates.

    Returns:
        np.ndarray: 1D list with neuron indices.
    """
    sorted_indices = np.argsort(
        coordinates[:, 0]
    )  # Sort by x-coordinate TODO build proper sorting
    return sorted_indices


def get_next_square_numbers(number):
    """
    Returns the two closest square numbers to the given number (larger square, smaller square).

    Args:
        number (int): The number for which to find the two closest square numbers.

    Returns:
        tuple: A tuple containing the two closest square numbers (larger square, smaller square).
    """

    square_ceil = int(np.ceil(np.sqrt(number))) ** 2
    square_floor = int(np.floor(np.sqrt(number))) ** 2

    if square_ceil == number:
        return square_ceil, square_ceil

    return square_ceil, square_floor


def convert_layer_size_to_grid_size(layer_size):
    """
    Converts a layer size to a (square) grid size.

    Args:
        layer_size (int): The number of neurons in the layer.

    Returns:
        tuple: A tuple containing the grid size as (rows, cols), rows==cols.
    """
    sqrt_layer_size = int(np.ceil(np.sqrt(layer_size)))
    # Verify that sqrt_layer_size^2 >= layer_size
    if sqrt_layer_size**2 < layer_size:
        raise ValueError("Layer size is not a square number")
    return sqrt_layer_size, sqrt_layer_size


def generate_square_grid_ascending(layer_size):
    """
    Generate a square grid of neuron indices in ascending order (for input layer).

    Args:
        layer_size (int): The number of neurons in the layer.

    Returns:
        np.ndarray: A 2D grid of neuron indices.
    """
    grid_size = convert_layer_size_to_grid_size(layer_size)
    grid = np.arange(layer_size).reshape(grid_size)
    return grid


def pad_array(array, target_size, pad_value=-1):
    """Do regular padding of a 2D array."""
    rows, cols = array.shape
    target_rows, target_cols = target_size

    # only operate if the first map is smaller
    if (rows, cols) == (target_rows, target_cols):
        return array
    elif rows > target_rows or cols > target_cols:
        raise ValueError("The array is larger than the target size.")

    new_array = np.full((target_rows, target_cols), pad_value)
    new_array[:rows, :cols] = array

    return new_array


def unpad_array(grid):
    """Unpad a grid by removing all -1 values."""
    grid = grid.flatten()
    grid = grid[grid != -1]
    grid = make_2d_grid_from_1d_list(grid)

    return grid


def intersperse_pad_array(array, target_size, pad_value=-1):
    """Intersperse padding of a 2D array."""
    rows, cols = array.shape
    target_rows, target_cols = target_size

    # only operate if the first map is smaller
    if (rows, cols) == (target_rows, target_cols):
        return array
    elif rows > target_rows or cols > target_cols:
        raise ValueError("The array is larger than the target size.")

    # Calculate where to place the original array values
    row_indices = np.linspace(0, target_rows - 1, rows, dtype=int)
    col_indices = np.linspace(0, target_cols - 1, cols, dtype=int)

    new_array = np.full((target_rows, target_cols), pad_value)
    for i, row in enumerate(row_indices):
        for j, col in enumerate(col_indices):
            new_array[row, col] = array[i, j]

    return new_array


def index_to_coord(index, grid_shape):
    """
    Convert a linear index to 2D coordinates (row, col) in a grid of given shape.

    Args:
        index (int): The linear index to be converted.
        grid_shape (tuple): The shape of the grid as (rows, cols).

    Returns:
        tuple: A tuple representing the 2D coordinates (row, col).
    """
    rows, cols = grid_shape
    return divmod(index, cols)


def manhattan_distance_2d(coord1, coord2):
    """
    Calculate the Manhattan distance in a 2D grid given two coordinates.

    Args:
        coord1 (tuple): The (row, col) coordinates of the first point.
        coord2 (tuple): The (row, col) coordinates of the second point.

    Returns:
        int: The Manhattan distance between the two points.
    """
    return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])


def euclidian_distance_2d(self, coord1, coord2):
    """
    Calculate the Euclidean distance between two points in 2D space.

    Args:
        coord1 (torch.Tensor): The coordinates of the first point.
        coord2 (torch.Tensor): The coordinates of the second point.

    Returns:
        torch.Tensor: The Euclidean distance between the two points.
    """
    return torch.sqrt(torch.sum((coord1 - coord2) ** 2))


def visualize_neuron_positions(net):
    """Visualize the 2D coordinates of the neurons in the network."""
    coordinates = net.lif1.coordinates.detach().numpy()
    plt.scatter(coordinates[:, 0], coordinates[:, 1])
    plt.show()


def visualize_neuron_positions_color(layer, spk_sum=None, title="Neuron Coordinates"):
    """Visualize the 2D coordinates of the neurons in the network with color-coded spike sums."""
    x_coords = layer.coordinates[:, 0].numpy()
    y_coords = layer.coordinates[:, 1].numpy()

    if spk_sum is not None:
        colors = spk_sum.numpy()
    else:
        colors = "b"  # Default color if spk_sum is not provided

    plt.scatter(x_coords, y_coords, c=colors, cmap="viridis", s=100, edgecolor="k")
    plt.colorbar(label="Spike Sum")
    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.show()


def visualize_heatmaps(heatmaps):
    """Visualize the values of the heatmaps as bar charts."""
    for heatmap in heatmaps:
        visualize_tensor(heatmap)


def visualize_tensor(tensor):
    """Visualize the values of a tensor as a bar chart."""
    # Convert tensor to numpy array
    values = tensor.detach().numpy()

    # Sort values for better visualization
    values = np.sort(values)

    # Create a bar chart
    plt.bar(range(len(values)), values)

    # Add labels and title
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("Tensor Values")

    # Show the plot
    plt.show()


def convert_tensors(obj):
    """
    Convert lists or dicts of tensors to lists of lists for JSON serialization.

    Args:
        obj (Union[torch.Tensor, list, dict]): The object to convert.

    Returns:
        Union[list, dict]: The converted object with tensors converted to lists.
    """
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_tensors(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_tensors(value) for key, value in obj.items()}
    else:
        return obj
