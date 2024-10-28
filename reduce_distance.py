# an algorithm to reduce the total manhattan distance between neurons in the entire network
import yaml
import numpy as np
from utils import make_2d_grid_from_1d_list, pad_array, unpad_array, intersperse_pad_array
from network_representation import NetworkRepresentation

def generate_rank_map(heatmap, epsilon=1e-5):
    """
    Generate a 2D rank map from the 1D input heatmap.
    """
    sorted_indices = np.argsort(heatmap)
    rank_map = np.zeros_like(heatmap, dtype=int)

    # Assign ranks with epsilon
    current_rank = 0
    for i in range(len(sorted_indices)):
        if (
            i > 0
            and abs(heatmap[sorted_indices[i]] - heatmap[sorted_indices[i - 1]])
            > epsilon
        ):
            current_rank += 1
        rank_map[sorted_indices[i]] = current_rank

    # Make square
    rank_map = make_2d_grid_from_1d_list(rank_map)

    return rank_map


def generate_rank_representation(config):
    # load file
    filepaths = config["filepaths"]
    network_representation_filepath = filepaths["network_representation_filepath"]
    with open(network_representation_filepath, "r") as file:
        network_representation = yaml.safe_load(file)

    rank_maps = []

    for heatmap in network_representation["heatmaps"]:
        rank_map = generate_rank_map(heatmap)
        rank_map = normalize_rank_map(rank_map)
        rank_maps.append(rank_map)

    layers = network_representation["layers"]

    for i in range(len(layers)):
        layers[i] = make_2d_grid_from_1d_list(layers[i])

    # TODO normalize ranks

    return rank_maps, layers


def normalize_rank_map(rank_map, padding_value=-1):
    """Normalize the rank map so that the maximum rank is len(rank_map) - 1."""
    flat_rank_map = rank_map.flatten()
    max_rank = np.max(flat_rank_map[flat_rank_map != padding_value])
    scaling_factor = (len(flat_rank_map) - 1) / max_rank

    # Apply the scaling factor to all non-padding values
    normalized_rank_map = np.where(
        rank_map != padding_value, rank_map * scaling_factor, padding_value
    )

    return normalized_rank_map


def compute_similarity_score(rank_map1, rank_map2):
    """0 means the rank maps are identical, negative values mean they are dissimilar."""
    return -1 * np.sum(np.abs(rank_map1 - rank_map2))


def compute_similarity_score_kernel(rank_map1, rank_map2, kernel_size=3):
    """Compute the similarity score between two rank maps using a kernel."""
    # use a kernel to compare the rank maps
    kernel = np.ones((kernel_size, kernel_size))
    kernel = kernel / np.sum(kernel)
    similarity_scores = []

    # pad to avoid edge cases
    rank_map1_padded = np.pad(rank_map1, kernel_size // 2, mode="constant")
    rank_map2_padded = np.pad(rank_map2, kernel_size // 2, mode="constant")

    # compare the rank maps with the kernel
    for i in range(rank_map1.shape[0]):
        for j in range(rank_map1.shape[1]):
            kernel1 = rank_map1_padded[i : i + kernel_size, j : j + kernel_size]
            kernel2 = rank_map2_padded[i : i + kernel_size, j : j + kernel_size]
            similarity_score = compute_similarity_score(kernel1, kernel2)
            similarity_scores.append(similarity_score)

    return np.mean(similarity_scores)


def consecutive_padding_amount(array, pad_value=-1):
    """Calculate the maximum number of consecutive padded values"""
    max_consecutive_pads = 0

    # Check rows for consecutive padded values
    for row in array:
        current_consecutive_pads = 0
        for value in row:
            if value == pad_value:
                current_consecutive_pads += 1
                if current_consecutive_pads > max_consecutive_pads:
                    max_consecutive_pads = current_consecutive_pads
            else:
                current_consecutive_pads = 0

    # Check columns for consecutive padded values
    for col in array.T:
        current_consecutive_pads = 0
        for value in col:
            if value == pad_value:
                current_consecutive_pads += 1
                if current_consecutive_pads > max_consecutive_pads:
                    max_consecutive_pads = current_consecutive_pads
            else:
                current_consecutive_pads = 0

    return max_consecutive_pads


def align_layer_sizes(grids):
    # get the size of the largest layer (assuming square arrays)
    max_layer_size = max([grid.shape[0] for grid in grids])

    # pad the smaller layers with -1
    padded_grids = []
    for grid in grids:
        padded_rank_map = pad_array(grid, (max_layer_size, max_layer_size))
        padded_grids.append(padded_rank_map)

    return padded_grids


def undo_align_layer_sizes(grids):
    # remove the padding from the layers
    unpadded_grids = []
    for grid in grids:
        unpadded_rank_map = unpad_array(grid)
        unpadded_grids.append(unpadded_rank_map)

    return unpadded_grids


def simulated_annealing(rank_map1, rank_map2, layer2, kernel_size=3):
    # TODO verify implementation
    # initialize the temperature
    temperature = 1.0
    temperature_min = 0.00001
    alpha = 0.95

    # initialize the current state
    current_state = rank_map2
    current_score = compute_similarity_score_kernel(rank_map1, rank_map2, kernel_size)

    # initialize the best state
    best_state = current_state
    best_score = current_score

    # Identify valid indices where rank_map2 values are not -1
    valid_indices2 = np.argwhere(rank_map2 != -1)

    while temperature > temperature_min:
        # generate a new state by shuffling the existing values (select two random positions and swap them)
        new_state = np.copy(current_state)
        """i1, j1 = np.random.randint(0, new_state.shape[0]), np.random.randint(
            0, new_state.shape[1]
        )
        i2, j2 = np.random.randint(0, new_state.shape[0]), np.random.randint(
            0, new_state.shape[1]
        )"""

        # Choose two different valid index positions to swap
        idx1, idx2 = np.random.choice(len(valid_indices2), size=2, replace=False)

        # Get the actual indices in rank_map2 from valid_indices2
        (i1, j1), (i2, j2) = valid_indices2[idx1], valid_indices2[idx2]

        new_state[i1, j1], new_state[i2, j2] = new_state[i2, j2], new_state[i1, j1]
        # also swap the values in the layer
        layer2[i1, j1], layer2[i2, j2] = layer2[i2, j2], layer2[i1, j1]

        # compute the new score
        new_score = compute_similarity_score_kernel(rank_map1, new_state, kernel_size)

        # accept the new state with a certain probability
        if new_score > current_score:
            current_state = new_state
            current_score = new_score
            if new_score > best_score:
                best_state = new_state
                best_score = new_score
        else:
            p = np.exp((new_score - current_score) / temperature)
            if np.random.rand() < p:
                current_state = new_state
                current_score = new_score

        # decrease the temperature
        temperature *= alpha

    return best_state


def reduce_distance(config):
    rank_maps, layers = generate_rank_representation(config)
    rank_maps = align_layer_sizes(rank_maps)
    layers = align_layer_sizes(layers)
    max_consecutive_pads = max(
        [consecutive_padding_amount(rank_map) for rank_map in rank_maps]
    )

    print("amount of rank maps:", len(rank_maps))
    print(
        "shapes of rank maps:",
        rank_maps[0].shape,
        rank_maps[1].shape,
        rank_maps[2].shape,
    )
    print("max consecutive pads:", max_consecutive_pads)
    for map in rank_maps:
        print(map)

    # reduce the distance between the layers
    for i in range(len(rank_maps) - 1):
        print("Simulated Annealing between layers", i, "and", i + 1)
        print(
            "Current score:", compute_similarity_score(rank_maps[i], rank_maps[i + 1])
        )
        print(
            "current score kernel:",
            compute_similarity_score_kernel(rank_maps[i], rank_maps[i + 1]),
        )
        rank_maps[i + 1] = simulated_annealing(
            rank_maps[i],
            rank_maps[i + 1],
            layers[i + 1],
            kernel_size=max_consecutive_pads,
        ) # Sorting of layers happens here
        print("New score:", compute_similarity_score(rank_maps[i], rank_maps[i + 1]))
        print(
            "new score kernel:",
            compute_similarity_score_kernel(rank_maps[i], rank_maps[i + 1]),
        )

    # unpad layers
    #rank_maps = undo_align_layer_sizes(rank_maps)
    layers = undo_align_layer_sizes(layers)

    # debugging: print the type of layers and the shape (should be list of lists)
    print(type(layers))


    return layers

def cluster_advanced(config):
    clustered_layers = reduce_distance(config)
    #clustered_layers = [layer.tolist() for layer in clustered_layers]

    filepath_unclustered = config["filepaths"]["network_representation_filepath"]
    filepath_clustered = config["filepaths"]["advanced_clustered_network_representation_filepath"]

    with open(filepath_unclustered) as file:
        network_unclustered = yaml.safe_load(file)

    weight_matrices = network_unclustered["weight_matrices"]
    heatmaps = network_unclustered["heatmaps"]

    network_representation = NetworkRepresentation(clustered_layers, weight_matrices, heatmaps)
    network_representation.export_representation(filepath_clustered)
    

if __name__ == "__main__":
    config_filepath = "data/config.yaml"
    with open(config_filepath, "r") as file:
        config = yaml.safe_load(file)

    rank_maps = reduce_distance(config)
