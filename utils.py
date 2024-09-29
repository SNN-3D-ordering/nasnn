import numpy as np
import torch
import matplotlib.pyplot as plt


def print_batch_accuracy(data, targets, net, batch_size, train=False):
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


def convert_layer_size_to_grid_size(layer_size):
    sqrt_layer_size = int(np.ceil(np.sqrt(layer_size)))
    # Verify that sqert_layer_size^2 >= layer_size
    if sqrt_layer_size * sqrt_layer_size < layer_size:
        raise ValueError("Layer size is not a square number")
    return sqrt_layer_size, sqrt_layer_size


def calculate_distance(self, coord1, coord2):
    # euclidean distance
    return torch.sqrt(torch.sum((coord1 - coord2) ** 2))


def visualize_neuron_positions(net):
    coordinates = net.lif1.coordinates.detach().numpy()
    plt.scatter(coordinates[:, 0], coordinates[:, 1])
    plt.show()
