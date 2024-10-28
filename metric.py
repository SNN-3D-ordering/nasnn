import json
from tqdm import tqdm
from utils import index_to_coord, manhattan_distance_2d
from utils import (
    convert_number_to_human_readable,
    convert_number_to_scientific_notation,
)
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
    heatmaps = data["heatmaps"]
    weight_matrices = data["weight_matrices"]  # TODO load only current one and next one
    layer_shapes = data["metadata"]["layer_dimensions"]

    current_layer = np.array(data["layers"][layer_idx]).flatten()
    next_layer = np.array(data["layers"][layer_idx + 1]).flatten()

    current_layer_heatmap = heatmaps[layer_idx]
    next_layer_heatmap = heatmaps[layer_idx + 1]
    weight_matrix = weight_matrices[layer_idx]

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

    # TODO instead of iterating over layer size, iterate over the current layer and take the values that are in there as indices for where to look in the heatmap
    # for n_idx in tqdm(
    #     range(num_neurons_current_layer),
    #     total=num_neurons_current_layer,
    #     desc=f"Layer {layer_idx} neurons",
    # ):
    #     if current_layer_heatmap[n_idx] == 0:
    #         continue  # Skip padded neurons in the current layer

    #     n_coord = index_to_coord(n_idx, current_shape)
    #     for m_idx in range(
    #         num_neurons_next_layer
    #     ):  # Loop through valid neurons in the next layer
    #         if next_layer_heatmap[m_idx] == 0:
    #             continue  # Skip padded neurons in the next layer

    #         m_coord = index_to_coord(m_idx, next_shape)

    #         # Adjust m_coord to align centers of different grid sizes
    #         adjusted_m_coord = (
    #             m_coord[0] + offset_current_to_next[0],
    #             m_coord[1] + offset_current_to_next[1],
    #         )

    #         # Get the weight from the matrix
    #         weight = (
    #             weight_matrix[m_idx][n_idx]
    #             if m_idx < len(weight_matrix) and n_idx < len(weight_matrix[m_idx])
    #             else 0
    #         )

    #         # Only consider non-zero weights
    #         if weight != 0:
    #             grid_distance = manhattan_distance_2d(n_coord, adjusted_m_coord)
    #             distance = (
    #                 1 + grid_distance
    #             )  # Vertical distance + grid distance, assume the vertical distance between layers is 1
    #             layer_distance += (
    #                 current_layer_heatmap[n_idx] * distance
    #             )  # Neurons that are more active contribute more to the distance

    # n_idx and m_idx are the values stored in the layer, which represent neuron indices.
    # n_grid_idx and m_grid_idx are the indices of those values in the layer.
    n_grid_idx = 0
    m_grid_idx = 0

    for n_idx in tqdm(
        current_layer,
        total=num_neurons_current_layer,
        desc=f"Layer {layer_idx} neurons",
    ):
        if current_layer_heatmap[n_idx] == 0:
            continue  # Skip padded neurons in the current layer

        n_coord = index_to_coord(n_grid_idx, current_shape)
        n_grid_idx += 1

        for m_idx in next_layer:  # Loop through valid neurons in the next layer
            if next_layer_heatmap[m_idx] == 0:
                continue  # Skip padded neurons in the next layer

            m_coord = index_to_coord(m_grid_idx, next_shape)
            m_grid_idx += 1
            # Adjust m_coord to align centers of different grid sizes
            adjusted_m_coord = (
                m_coord[0] + offset_current_to_next[0],
                m_coord[1] + offset_current_to_next[1],
            )

            # Get the weight from the matrix
            weight = (
                weight_matrix[m_idx][n_idx]
                if m_idx < len(weight_matrix) and n_idx < len(weight_matrix[m_idx])
                else 0
            )

            # Only consider non-zero weights
            if weight != 0:
                grid_distance = manhattan_distance_2d(n_coord, adjusted_m_coord)
                distance = (
                    1 + grid_distance
                )  # Vertical distance + grid distance, assume the vertical distance between layers is 1
                layer_distance += (
                    current_layer_heatmap[n_idx] * distance
                )  # Neurons that are more active contribute more to the distance

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
    print(
        f"Total weighted Manhattan distance for the entire network: \n {total_distance} (= {convert_number_to_human_readable(total_distance)} = {convert_number_to_scientific_notation(total_distance)})"
    )
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
