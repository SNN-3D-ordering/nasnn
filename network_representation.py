import json


class NetworkRepresentation:
    def __init__(self, layers=[]):
        self.layers = (
            layers  # List of layers, each layer is a 2D array of neuron indices
        )
        self.layer_connections = [
            {} for _ in layers
        ]  # List of dicts to store connections for each layer

    def add_layer(self, layer):
        self.layers.append(layer)
        self.layer_connections.append({})

    def add_connection(self, layer_idx, neuron_i, neuron_j):
        connection = (neuron_i, neuron_j)
        if connection in self.layer_connections[layer_idx]:
            self.layer_connections[layer_idx][connection] += 1
        else:
            self.layer_connections[layer_idx][connection] = 1

    def export_representation(self, file_path):
        representation = {
            "layers": [layer.tolist() for layer in self.layers],
            "layer_connections": self.layer_connections,
        }
        with open(file_path, "w") as json_file:
            json.dump(representation, json_file, indent=4)
