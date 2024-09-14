import json


class NetworkRepresentation:
    def __init__(self, layers=[]):
        self.layers = (
            layers  # List of layers, each layer is a 2D array of neuron indices
        )
        self.connections = {}  # Dictionary to store connections

    def add_layer(self, layer):
        self.layers.append(layer)

    def add_connection(self, neuron_i, neuron_j):
        connection = (neuron_i, neuron_j)
        if connection in self.connections:
            self.connections[connection] += 1
        else:
            self.connections[connection] = 1

    def export_representation(self, file_path):
        representation = {
            "layers": [layer.tolist() for layer in self.layers],
            "connections": self.connections,
        }
        with open(file_path, "w") as json_file:
            json.dump(representation, json_file, indent=4)
