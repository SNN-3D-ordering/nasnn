import json

# TODO: Heatmaps/ wann feuert welches neuron?
# TODO: Array übertragen in json
# TODO: weight matrices ==layer_connections in NetworkRepresentation


class NetworkRepresentation:
    def __init__(self, layers=[], weight_matrices=[], heatmaps=[]):
        self.layers = (
            layers  # List of spiking layers, each layer is a 2D array of neuron indices
        )
        self.weight_matrices = weight_matrices  # List of weight matrices for each fully connected layer
        self.heatmaps = heatmaps  # Initialize heatmaps for each layer
        self.activations = []  # List of activations for each layer (batch_amount lists of layer_size activations)

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
        print(len(self.layers))
        print(len(self.weight_matrices))
        print(len(self.heatmaps))
        assert len(self.layers) == len(self.heatmaps)
        assert len(self.layers) - 1 == len(self.weight_matrices)

        metadata = {
            "num_layers": len(self.layers),
            "layer_dimensions": [layer.shape for layer in self.layers],
        }

        # Convert tensors to lists for JSON serialization
        heatmap_lists = [
            heatmap.tolist() if heatmap is not None else None
            for heatmap in self.heatmaps
        ]

        weight_matrix_lists = [
            weight_matrix.tolist() if weight_matrix is not None else None
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
        assert len(self.layers) == len(self.activations[1])

        metadata = {
            "num_layers": self.activations[1],
            "layer_dimensions": [layer.shape for layer in self.layers],
            "batch_amount": self.activations[0],
        }

        # Convert activations to lists for JSON serialization. 
        activation_lists = [
            [activation.tolist() for activation in batch]
            for batch in self.activations
        ]

        representation = {
            "metadata": metadata,
            "layers": [layer.tolist() for layer in self.layers],
            "activations": activation_lists,
        }

        with open(file_path, "w") as json_file:
            json.dump(representation, json_file, indent=4)