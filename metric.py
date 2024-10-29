import json
from tqdm import tqdm
from utils import index_to_coord, manhattan_distance_2d
from utils import (
    convert_number_to_human_readable,
    convert_number_to_scientific_notation,
)
from utils import visualize_heatmap, visualize_layer_with_heatmap
import numpy as np


def calculate_distance_between_layers(data, layer_idx):
    """
    Calculate the weighted Manhattan distance between two consecutive layers in a network.

    Args:
        data (dict): The network data loaded from the JSON file, containing heatmaps, weight
                     matrices, and layer dimensions.
        layer_idx (int): The index of the current layer for which to calculate the distance
                         to the next layer.

    Returns:
        int: The computed weighted Manhattan distance between the current layer and the next layer.

    Raises:
        ValueError: If the dimensions of the weight matrix do not match the number of neurons in the
                    current and next layers.
    """
    # Load all data
    layer_shapes = data["metadata"]["layer_dimensions"]

    current_layer = np.array(data["layers"][layer_idx]).flatten()
    next_layer = np.array(data["layers"][layer_idx + 1]).flatten()

    current_layer_heatmap = data["heatmaps"][layer_idx]
    next_layer_heatmap = data["heatmaps"][layer_idx + 1]

    weight_matrix = data["weight_matrices"][layer_idx]

    #visualize_layer_with_heatmap(
    #    current_layer, current_layer_heatmap, title=f"Layer {layer_idx}"
    #)
    current_shape = layer_shapes[layer_idx]
    next_shape = layer_shapes[layer_idx + 1]

    num_neurons_current_layer = len(current_layer_heatmap)
    num_neurons_next_layer = len(next_layer_heatmap)

    # Check that the shapes match, if not raise an error
    if (
        len(weight_matrix) != num_neurons_next_layer
        or len(weight_matrix[0]) != num_neurons_current_layer
    ):
        raise ValueError(
            f"Weight matrix dimensions do not match the number of neurons in the current and next layers. "
            f"Expected: ({num_neurons_next_layer}, {num_neurons_current_layer}), "
            f"but got: ({len(weight_matrix)}, {len(weight_matrix[0]) if weight_matrix else 0})"
        )

    # Calculate offsets to center grids in order to align the layers by their centers
    offset_current_to_next = (
        (current_shape[0] - next_shape[0]) // 2,
        (current_shape[1] - next_shape[1]) // 2,
    )

    layer_distance = 0

    # Measure the distance between each pair of neurons in the current and next layers
    for i in tqdm(
        range(num_neurons_current_layer),
        total=num_neurons_current_layer,
        desc=f"Layer {layer_idx} neurons",
    ):
        # Translate the index to the heatmap index, get the heat value
        n_heatmap_idx = current_layer[i]
        n_heat = current_layer_heatmap[n_heatmap_idx]
        if n_heat == 0:
            continue # This skips both padded and dead neurons
        n_coord = index_to_coord(i, current_shape)

        for j in range(num_neurons_next_layer):
            # idx -> heatmap index for the next layer too
            m_heatmap_idx = next_layer[j]
            m_heat = next_layer_heatmap[m_heatmap_idx]
            if m_heat == 0:
                continue

            # Adjust the coordinates of the next layer to align the centers
            m_coord = index_to_coord(j, next_shape)
            m_coord = (
                m_coord[0] + offset_current_to_next[0],
                m_coord[1] + offset_current_to_next[1],
            )

            # Add distance between two neurons to layer distance if a signal gets transmitted
            weight = weight_matrix[j][i]
            if weight != 0:
                grid_distance = manhattan_distance_2d(n_coord, m_coord)
                distance = 1 + grid_distance # Assumption that vertical distance is 1
                layer_distance += n_heat * distance # Weigh by activity of the neuron

    return layer_distance


def calculate_total_distance(data):
    """
    Calculate the total weighted Manhattan distance for the entire network.

    Args:
        data (dict): The network data loaded from the JSON file, containing heatmaps,
                     weight matrices, and layer dimensions.

    Returns:
        int: The total weighted Manhattan distance for the entire network.
    """
    num_layers = data["metadata"]["num_layers"]
    total_distance = 0
    layer_distances = []

    # Iterate through each layer except the last one
    for layer_idx in range(num_layers - 1):
        layer_distance = calculate_distance_between_layers(data, layer_idx)
        layer_distances.append(layer_distance)

    print(f"Layer distances: {layer_distances}")

    total_distance = sum(layer_distances)

    return total_distance


def measure_network(filepath):
    """
    Measure the total weighted Manhattan distance for the entire network.

    Args:
        filepath (str): The path to the JSON file containing the network representation.

    Returns:
        int: The total weighted Manhattan distance for the entire network.
    """
    # Load the JSON data from the file
    with open(filepath, "r") as f:
        data = json.load(f)

    # Calculate the total Manhattan distance for the network
    total_distance = calculate_total_distance(data)

    return total_distance


if __name__ == "__main__":
    total_distance = measure_network("network_representation.json")
    print(
        f"Total weighted Manhattan distance for the entire network: {total_distance} (= {convert_number_to_human_readable(total_distance)} = {convert_number_to_scientific_notation(total_distance)})"
    )
