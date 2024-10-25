# an algorithm to reduce the total manhattan distance between neurons in the entire network
import yaml
import numpy as np
from utils import get_next_square_numbers

config_filepath = "data/config.yaml"
with open(config_filepath, "r") as file:
    config = yaml.safe_load(file)


def generate_rank_representation(config):
    filepaths = config["filepaths"]
    network_representation_filepath = filepaths["network_representation_filepath"]
    with open(network_representation_filepath, "r") as file:
        network_representation = yaml.safe_load(file)

    rank_maps = []
    # for the input layer:
    input_layer = network_representation["layers"][0]
    input_heatmap = input_layer["heatmap"]
    input_rank_map = np.argsort(np.argsort(input_heatmap, axis=None)).reshape(
        input_heatmap.shape
    )
    rank_maps.append(input_rank_map)

    # for each further layer, use the non-0 weights as mask TODO verify functionality
    for layer in network_representation["layers"][1:]:
        weight_matrix = layer["weight_matrix"]
        weight_matrix_binary = (weight_matrix != 0).astype(int)
        heatmap = np.multiply(input_heatmap, weight_matrix_binary)
        rank_map = np.argsort(np.argsort(heatmap, axis=None)).reshape(heatmap.shape)
        rank_maps.append(rank_map)

    return rank_maps


def compute_similarity_score(rank_map1, rank_map2):
    return -1 * np.sum(np.abs(rank_map1 - rank_map2))


def compute_similarity_score_kernel(rank_map1, rank_map2, kernel_size=3):
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


def intersperse_pad_array(array, target_size, pad_value=-1):
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


def align_layer_sizes(rank_maps):
    # get the size of the largest layer
    max_layer_size = max([rank_map.size for rank_map in rank_maps])

    # pad the smaller layers with -1
    padded_rank_maps = []
    for rank_map in rank_maps:
        padded_rank_map = intersperse_pad_array(
            rank_map, (max_layer_size, max_layer_size)
        )
        padded_rank_maps.append(padded_rank_map)

    return padded_rank_maps


def unpad_layer(rank_map):
    # remove the -1 padding
    rank_map = rank_map[rank_map != -1]
    # add new -1s to make the rank_map square
    next_square = get_next_square_numbers(rank_map.size)[0]
    rank_map = np.pad(
        rank_map, (0, next_square - rank_map.size), mode="constant", constant_values=-1
    )


def simulated_annealing(rank_map1, rank_map2, kernel_size=3):
    # TODO verify implementation
    # initialize the temperature
    temperature = 1.0
    temperature_min = 0.00001
    alpha = 0.9

    # initialize the current state
    current_state = rank_map2
    current_score = compute_similarity_score_kernel(rank_map1, rank_map2, kernel_size)

    # initialize the best state
    best_state = current_state
    best_score = current_score

    while temperature > temperature_min:
        # generate a new state
        new_state = np.copy(current_state)
        i = np.random.randint(0, new_state.shape[0])
        j = np.random.randint(0, new_state.shape[1])
        new_state[i, j] = np.random.randint(0, new_state.size)

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
    rank_maps = generate_rank_representation(config)
    rank_maps = align_layer_sizes(rank_maps)

    # reduce the distance between the layers
    for i in range(len(rank_maps) - 1):
        rank_maps[i + 1] = simulated_annealing(rank_maps[i], rank_maps[i + 1])

    # unpad layers
    for i in range(len(rank_maps)):
        rank_maps[i] = unpad_layer(rank_maps[i])

    return rank_maps


if __name__ == "__main__":
    rank_maps = reduce_distance(config)
    print(rank_maps)
