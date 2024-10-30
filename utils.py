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


def make_grid_from_coordinates_radius_based(coordinates):
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

    grid_size = convert_layer_size_to_grid_size(len(coordinates))
    rows, cols = grid_size
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
    grid_x = np.linspace(min_x, max_x, cols + 1)
    grid_y = np.linspace(min_y, max_y, rows + 1)

    # Initialize a grid array with -1 indicating empty cells
    grid = np.full((rows, cols), -1)

    # List to store the map of coordinates to grid locations
    mapped_coords = []

    # Calculate the halfway point of the grid size
    half_rows = rows / 2
    half_cols = cols / 2

    # Loop through the indices of points sorted by proximity to the centroid
    for idx in sorted_indices:
        point = coordinates[idx]

        # Compute initial grid cell coordinates based on the transformed position about the centroid
        rel_x = min(
            max(
                int((point[0] - centroid[0]) / ((max_x - min_x) / cols) + half_cols),
                0,
            ),
            cols - 1,
        )
        rel_y = min(
            max(
                int((point[1] - centroid[1]) / ((max_y - min_y) / rows) + half_rows),
                0,
            ),
            rows - 1,
        )

        # Attempt to place it at calculated cell, else find nearest free cell
        placed = False

        # Loop over the range of possible offsets (distance from the initial target cell)
        for offset in range(max(rows, cols)):
            # Iterate over horizontal offsets (-offset to +offset)
            for dx in range(-offset, offset + 1):
                # Iterate over vertical offsets (-offset to +offset)
                for dy in range(-offset, offset + 1):
                    row_idx, col_idx = (rel_y + dy) % rows, (rel_x + dx) % cols

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


def make_grid_from_coordinates_linear(coordinates, pad_value=-1):
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
        print("Converting layer size to grid size: Layer size is not a square number")
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


def visualize_heatmap(heatmap, title="Heatmap"):
    # a function that takes a 1d heatmap and visualizes it as a 2d grid
    heatmap = np.array(heatmap)
    # normalize heatmap
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    # make heatmap square
    heatmap = make_2d_grid_from_1d_list(heatmap)

    # visualize heatmap
    plt.imshow(heatmap, cmap="hot")
    plt.colorbar()
    plt.title(title)
    plt.show()


def visualize_layer_with_heatmap(layer, heatmap, title="Layer with Heatmap"):
    # iterate over the layer grid and set the value to the corresponding value in the heatmap
    heatmap = np.array(heatmap)
    heatmap = (heatmap - heatmap.min()) / (
        heatmap.max() - heatmap.min()
    )  # TODO ignore -1

    aligned_heatmap = np.zeros_like(layer, dtype=float)
    # iterate over the layer
    for i in range(layer.size):
        aligned_heatmap[i] = heatmap[layer[i]]

    # make layer square
    aligned_heatmap = make_2d_grid_from_1d_list(aligned_heatmap)

    # visualize layer with heatmap
    plt.imshow(aligned_heatmap, cmap="hot")
    plt.colorbar()
    plt.title(title)
    plt.show()


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


def convert_number_to_human_readable(number):
    """
    Convert a number to a human-readable format.

    Args:
        number (int): The number to convert.

    Returns:
        str: The human-readable number.
    """
    number = float(number)

    if abs(number) < 1e3:
        return str(number)
    elif abs(number) < 1e6:
        return f"{number/1e3:.1f}K"
    elif abs(number) < 1e9:
        return f"{number/1e6:.1f}M"
    else:
        return f"{number/1e9:.1f}B"


def convert_number_to_scientific_notation(number):
    """
    Convert a number to scientific notation.

    Args:
        number (int): The number to convert.

    Returns:
        str: The number in scientific notation.
    """
    return "{:.2e}".format(number)


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


def prune_weights(network, threshold):
    """
    Set all weights in the network that are below the threshold to zero.

    Args:
        network (nn.Module): The neural network to prune.
        threshold (float): The threshold below which weights are set to zero.
    """
    for name, param in network.named_parameters():
        if "weight" in name:
            with torch.no_grad():
                # Create a mask of weights below the threshold
                mask = param.abs() < threshold
                # Set weights below the threshold to zero
                param[mask] = 0
