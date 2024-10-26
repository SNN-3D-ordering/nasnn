import torch
import yaml
from model import Net
from data import get_data_loaders
from train import train_model
from test import test, cluster_simple, record
from utils import get_next_square_numbers
import matplotlib.pyplot as plt
import argparse
from network_representation import NetworkRepresentation
from metric import measure_network

# Argument parser for command line flags
parser = argparse.ArgumentParser(description="Train or evaluate the model.")
parser.add_argument("--train", "-t", action="store_true", help="Flag to run training")
parser.add_argument("--eval", "-e", action="store_true", help="Flag to run evaluation")
parser.add_argument(
    "--model-path", type=str, default="model.pth", help="Path to save/load the model"
)
args = parser.parse_args()

config_filepath = "data/config.yaml"
with open(config_filepath, "r") as file:
    config = yaml.safe_load(file)

model_params = config["model"]
training_params = config["training"]

# verify that num_inputs, num_hidden are square numbers
bigger_num_inputs, smaller_num_inputs = get_next_square_numbers(
    model_params["num_inputs"]
)
bigger_num_hidden, smaller_num_hidden = get_next_square_numbers(
    model_params["num_hidden"]
)

if (
    bigger_num_inputs != model_params["num_inputs"]
    or bigger_num_hidden != model_params["num_hidden"]
):
    print(
        f"Some model parameters were not square numbers. Consider adjusting to the next bigger or smaller square numbers to avoid padding:"
    )
    if bigger_num_inputs != model_params["num_inputs"]:
        print(
            f"num_inputs: {model_params['num_inputs']} -> {bigger_num_inputs} or {smaller_num_inputs}"
        )

    if bigger_num_hidden != model_params["num_hidden"]:
        print(
            f"num_hidden: {model_params['num_hidden']} -> {bigger_num_hidden} or {smaller_num_hidden}"
        )


# Initialize data loaders
train_loader, test_loader = get_data_loaders(
    training_params["batch_size"], training_params["data_path"]
)

# Initialize model
# if torch.cuda.is_available():
#    device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#    device = torch.device("mps")
# else:
#    device = torch.device("cpu")

# Use CPU for now (this is because the indices and tensors are not on the same device) TODO fix this
device = torch.device("cpu")
net = Net(
    model_params["num_inputs"],
    model_params["num_hidden"],
    model_params["num_outputs"],
    model_params["num_steps"],
    model_params["beta"],
).to(device)

if args.train:
    # Train model
    loss_hist, test_loss_hist = train_model(
        net,
        train_loader,
        test_loader,
        device,
        model_params["num_steps"],
        training_params["num_epochs"],
        training_params["batch_size"],
        torch.float,
    )

    # Save the trained model
    torch.save(net.state_dict(), config["filepaths"]["model_filepath"])

    # Plot the loss curves
    fig = plt.figure(facecolor="w", figsize=(10, 5))
    plt.plot(loss_hist)
    plt.plot(test_loss_hist)
    plt.title("Loss Curves")
    plt.legend(["Train Loss", "Test Loss"])
    plt.show()

if args.eval:
    # Load the trained model
    net.load_state_dict(torch.load(config["filepaths"]["model_filepath"]))

    # Evaluate model
    total, correct = test(net, test_loader, device, config)
    print(f"Accuracy: {correct/total*100:.2f}%")

    # Measure network
    total_distance = measure_network(config["filepaths"]["network_representation_filepath"])
    print(f"Total weighted Manhattan distance for the entire network: {total_distance}")

    # Run simple clustering
    print("Clustering...")
    cluster_simple(net, test_loader, device, config, max_steps=1000)

    # Measure network again
    total_distance = measure_network(config["filepaths"]["network_representation_filepath"])
    print(f"Total weighted Manhattan distance for the entire network after clustering: {total_distance}")

    # Record spike times for 100 batches
    # print("Recording spike times...")
    # record(net, test_loader, device, config, record_batches=10)

if not args.train and not args.eval:
    print("Please specify either --train/-t or --eval/-e flag.")
