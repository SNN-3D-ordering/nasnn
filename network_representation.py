import json

# TODO: Heatmaps/ wann feuert welches neuron?
# TODO: Array übertragen in json
# TODO: weight matrices ==layer_connections in NetworkRepresentation


class NetworkRepresentation:
    def __init__(self, layers=[]):
        self.layers = (
            layers  # List of layers, each layer is a 2D array of neuron indices
        )
        self.weight_matrices = [
            {} for _ in layers
        ]  # List of dicts to store connections for each layer
        self.heatmaps = [None for _ in layers]  # Initialize heatmaps for each layer

    def add_layer(self, layer):
        self.layers.append(layer)
        self.weight_matrices.append({})
        self.heatmaps.append(None)

    def add_connection(self, layer_idx, neuron_i, neuron_j):
        connection = (neuron_i, neuron_j)
        if connection in self.weight_matrices[layer_idx]:
            self.weight_matrices[layer_idx][connection] += 1
        else:
            self.weight_matrices[layer_idx][connection] = 1

    def make_heatmap(self, spike_counts, layer_idx):
        """Function that creates a heatmap for a given layer based on spike counts"""
        if layer_idx >= len(self.layers):
            print(f"No layer found at index {layer_idx}.")
            return

        neuron_grid = self.layers[layer_idx]
        heatmap_data = spike_counts.reshape(neuron_grid.shape)

        # sns.heatmap(heatmap_data.cpu(), annot=True, fmt="d", cmap="YlGnBu")
        # plt.title(f"Neuron firing heatmap for layer {layer_idx}")

    def export_representation(self, file_path):
        # Convert heatmaps to lists for JSON serialization
        heatmap_lists = [
            heatmap.tolist() if heatmap is not None else None
            for heatmap in self.heatmaps
        ]

        representation = {
            "layers": [layer.tolist() for layer in self.layers],
            "weight_matrices": self.weight_matrices,
            "heatmaps": heatmap_lists,
        }
        with open(file_path, "w") as json_file:
            json.dump(representation, json_file, indent=4)
