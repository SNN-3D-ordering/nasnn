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


def map_to_2d_grid_row_wise(coordinates, grid_size):
    """
    Maps 2D coordinates to a 2D grid.

    Args:
        coordinates (np.ndarray): Array of 2D coordinates.
        grid_size (tuple): Size of the grid (rows, cols).

    Returns:
        np.ndarray: 2D grid with neuron indices.
    """
    num_neurons = coordinates.shape[0]
    sorted_indices = np.argsort(coordinates[:, 0])  # Sort by x-coordinate
    grid = np.zeros(grid_size, dtype=int) - 1  # Initialize grid with -1 (empty)

    for idx, neuron_idx in enumerate(sorted_indices):
        row = idx // grid_size[1]
        col = idx % grid_size[1]
        grid[row, col] = neuron_idx

    return grid


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
