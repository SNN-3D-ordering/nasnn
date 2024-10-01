import json


# Function to load the network representation from a JSON file
def load_network_representation(file_path):
    with open(file_path, "r") as json_file:
        representation = json.load(json_file)
    return representation


# Function to inspect the input layer of the network
def inspect_input_layer(network_representation):
    input_layer = network_representation["layers"][0]
    layer_shape = network_representation["metadata"]["layer_dimensions"][0]
    print("Input Layer Shape:", layer_shape)
    print("Input Layer Data:")
    for row in input_layer:
        print(row)


if __name__ == "__main__":
    file_path = "network_representation.json"  # Update this if your file has a different name or location

    # Load the network representation from the JSON file
    network_representation = load_network_representation(file_path)

    # Inspect the input layer
    inspect_input_layer(network_representation)
