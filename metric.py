import json
from tqdm import tqdm


def index_to_coord(index, grid_shape):
    """Convert a linear index to 2D coordinates (row, col) in a grid of given shape."""
    rows, cols = grid_shape
    return divmod(index, cols)


def manhattan_distance_2d(coord1, coord2):
    """Calculate the Manhattan distance in a 2D grid given two coordinates."""
    return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])


def calculate_distance_between_layers(data, layer_idx):
    heatmaps = data["heatmaps"]
    weight_matrices = data["weight_matrices"]
    layer_shapes = data["metadata"]["layer_dimensions"]

    current_layer_heatmap = heatmaps[layer_idx]
    next_layer_heatmap = heatmaps[layer_idx + 1]
    weight_matrix = weight_matrices[layer_idx]

    current_shape = layer_shapes[layer_idx]
    next_shape = layer_shapes[layer_idx + 1]

    num_neurons_current_layer = len(current_layer_heatmap)
    num_neurons_next_layer = len(next_layer_heatmap)

    print(f"Processing between layer {layer_idx} and layer {layer_idx + 1}:")
    print(f"Current layer shape: {current_shape}, Next layer shape: {next_shape}")
    print(
        f"Heatmap length (current layer): {len(current_layer_heatmap)}, (next layer): {len(next_layer_heatmap)}"
    )
    print(
        f"Weight matrix shape: ({len(weight_matrix)}, {len(weight_matrix[0]) if weight_matrix else 0})"
    )

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

    # Calculate offsets to center grids
    offset_current_to_next = (
        (current_shape[0] - next_shape[0]) // 2,
        (current_shape[1] - next_shape[1]) // 2,
    )

    layer_distance = 0

    for n_idx in tqdm(
        range(num_neurons_current_layer),
        total=num_neurons_current_layer,
        desc=f"Layer {layer_idx} neurons",
    ):
        if current_layer_heatmap[n_idx] == 0:
            continue  # Skip padded neurons in the current layer

        n_coord = index_to_coord(n_idx, current_shape)
        for m_idx in range(
            num_neurons_next_layer
        ):  # Loop through valid neurons in the next layer
            if next_layer_heatmap[m_idx] == 0:
                continue  # Skip padded neurons in the next layer

            m_coord = index_to_coord(m_idx, next_shape)

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
                layer_distance += current_layer_heatmap[n_idx] * distance

    return layer_distance


def calculate_total_distance(data):
    num_layers = data["metadata"]["num_layers"]
    total_distance = 0

    # Iterate through each layer except the last one
    for layer_idx in range(num_layers - 1):
        layer_distance = calculate_distance_between_layers(data, layer_idx)
        print(
            f"Total weighted Manhattan distance between layer {layer_idx} and layer {layer_idx + 1}: {layer_distance}"
        )
        total_distance += layer_distance

    return total_distance


# Load the JSON data from the file
with open("network_representation.json", "r") as f:
    data = json.load(f)

# Calculate the total Manhattan distance for the network
total_distance = calculate_total_distance(data)
print(f"Total weighted Manhattan distance for the entire network: {total_distance}")
