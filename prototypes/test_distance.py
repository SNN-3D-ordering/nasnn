import numpy as np
from tqdm import tqdm


def index_to_coord(index, shape):
    """Convert a linear index to a 2D coordinate based on the given shape."""
    return divmod(index, shape[1])


def manhattan_distance_2d(coord1, coord2):
    """Calculate the Manhattan distance between two 2D coordinates."""
    return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])


def debug_calculate_distance():
    """
    Calculate the weighted Manhattan distance between two manually defined layers for debugging purposes.
    """
    # Manually defined heatmaps and weight matrix
    current_layer_heatmap = np.array([1, 0, 0, 100])
    next_layer_heatmap = np.array([1, 0, 0, 0])
    weight_matrix = np.array(
        [
            [0.5, 0.2, 0.0, 0.1],
            [0.3, 0.0, 0.4, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.1, 0.2, 0.3, 0.4],
        ]
    )

    current_shape = (2, 2)  # 2x2 grid
    next_shape = (2, 2)  # 2x2 grid

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

    for n_idx in tqdm(
        range(num_neurons_current_layer),
        total=num_neurons_current_layer,
        desc=f"Layer neurons",
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
                layer_distance += (
                    current_layer_heatmap[n_idx] * distance
                )  # Neurons that are more active contribute more to the distance

    return layer_distance


# Call the debug function
distance = debug_calculate_distance()
print(f"Calculated distance: {distance}")
