import json
import os

# Function to convert the JSON structure
def convert_structure(data):
    converted = {"layers": []}

    for layer_index, layer in enumerate(data["layers"]):
        layer_data = {"neurons": []}
        heatmap = data["heatmaps"][layer_index]

        for row_index, row in enumerate(layer):
            for col_index, neuron_id in enumerate(row):
                heat_value = heatmap[neuron_id]
                neuron_data = {
                    "id": neuron_id,
                    "position": [col_index, row_index],
                    "heat": heat_value
                }
                layer_data["neurons"].append(neuron_data)

        converted["layers"].append(layer_data)

    return converted

def load_and_convert_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    # Convert structure using previously defined function
    return convert_structure(data)

# Converting the structure
# Load the JSON file
filename = "exports/simple_clustered_network_representation.json"
data = load_and_convert_json(filename)

# Save the converted structure to a new JSON file
converted_filename = "exports/converted_structure.json"
with open(converted_filename, 'w') as file:
    json.dump(data, file, indent=4)