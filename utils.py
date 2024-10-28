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
    Maps 2D coordinates to a 2D grid, starting from the center and filling in a spiral.

    Args:
        coordinates (np.ndarray): Array of 2D coordinates.
        grid_size (tuple): Size of the grid (rows, cols).

    Returns:
        np.ndarray: 2D grid with neuron indices indexed spirally from the center.
    """
    num_neurons = coordinates.shape[0]

    # Calculate the center of the coordinates
    center = np.mean(coordinates, axis=0)

    # Calculate distance from center for each point and sort based on it
    distances = np.linalg.norm(coordinates - center, axis=1)
    sorted_indices = np.argsort(distances)

    # Initialize grid with -1 (empty)
    grid = np.zeros(grid_size, dtype=int) - 1

    # Determine the center of the grid
    mid_row = grid_size[0] // 2
    mid_col = grid_size[1] // 2

    # Directions for spiral (right, down, left, up)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    dir_idx = 0  # Start with the 'right' direction

    # Initialize the starting point in the center of the grid
    row, col = mid_row, mid_col

    # Place neurons in the grid following the spiral order
    for idx in sorted_indices:
        grid[row, col] = idx
        next_row = row + directions[dir_idx][0]
        next_col = col + directions[dir_idx][1]

        # Check if the new position is within the grid and empty; else, change direction
        if not (
            0 <= next_row < grid_size[0]
            and 0 <= next_col < grid_size[1]
            and grid[next_row, next_col] == -1
        ):
            dir_idx = (dir_idx + 1) % 4  # Change direction
            next_row = row + directions[dir_idx][0]
            next_col = col + directions[dir_idx][1]

        # Move to the next cell in the spiral
        row, col = next_row, next_col

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
