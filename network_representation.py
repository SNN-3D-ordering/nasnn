import json
from utils import convert_tensors

# TODO: Heatmaps/ wann feuert welches neuron?
# TODO: Array übertragen in json
# TODO: weight matrices ==layer_connections in NetworkRepresentation


class NetworkRepresentation:
    def __init__(self, layers=[], weight_matrices=[], heatmaps=[]):
        self.layers = (
            layers  # List of spiking layers, each layer is a 2D array of neuron indices
        )
        self.weight_matrices = (
            weight_matrices  # List of weight matrices for each fully connected layer
        )
        self.heatmaps = heatmaps  # Initialize heatmaps for each layer
        self.activations = (
            []
        )  # List of activations for each layer (batch_amount lists of layer_size activations)

    def add_layer(self, layer, layer_idx=None):
        if layer_idx is None:
            self.layers.append(layer)
        else:
            self.layers[layer_idx] = layer

    def add_weight_matrix(self, weight_matrix, layer_idx=None):
        if layer_idx is None:
            self.weight_matrices.append(weight_matrix)
        else:
            self.weight_matrices[layer_idx] = weight_matrix

    def add_heatmap(self, heatmap, layer_idx=None):
        if layer_idx is None:
            self.heatmaps.append(heatmap)
        else:
            self.heatmaps[layer_idx] = heatmap

    # def add_connection(self, layer_idx, neuron_i, neuron_j):
    #     connection = (neuron_i, neuron_j)
    #     if connection in self.weight_matrices[layer_idx]:
    #         self.weight_matrices[layer_idx][connection] += 1
    #     else:
    #         self.weight_matrices[layer_idx][connection] = 1

    def export_representation(self, file_path):
        # Assert that the amounts match up
        assert len(self.layers) == len(self.heatmaps)
        assert len(self.layers) - 1 == len(self.weight_matrices)

        metadata = {
            "num_layers": len(self.layers),
            "layer_dimensions": [layer.shape for layer in self.layers],
        }

        # Convert tensors to lists for JSON serialization, if they are not already lists
        heatmap_lists = [
            heatmap.tolist() if not isinstance(heatmap, list) and heatmap is not None else heatmap
            for heatmap in self.heatmaps
        ]
        
        weight_matrix_lists = [
            weight_matrix.tolist() if not isinstance(weight_matrix, list) and weight_matrix is not None else weight_matrix
            for weight_matrix in self.weight_matrices
        ]
        
        layer_lists = [layer.tolist() for layer in self.layers]

        representation = {
            "metadata": metadata,
            "layers": layer_lists,
            "weight_matrices": weight_matrix_lists,
            "heatmaps": heatmap_lists,
        }

        with open(file_path, "w") as json_file:
            json.dump(representation, json_file, indent=4)

    def export_activations(self, activations, file_path):
        # Assert that the amounts match up
        assert len(self.layers) == len(activations[0])

        metadata = {
            "num_layers": len(activations[0]),
            "layer_dimensions": [layer.shape for layer in self.layers],
            "batch_amount": len(activations),
        }

        representation = {
            "metadata": metadata,
            "layers": [layer.tolist() for layer in self.layers],
            "activations": convert_tensors(activations),
        }

        with open(file_path, "w") as json_file:
            json.dump(representation, json_file, indent=4)
