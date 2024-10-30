# grid_search.py

import torch
import yaml
import argparse
from big_model import Net
from data import get_data_loaders
from train import train_model
from test import test
from utils import get_next_square_numbers, prune_weights
import matplotlib.pyplot as plt

# Argument parser for command line options
parser = argparse.ArgumentParser(
    description="Grid Search over Pruning and LIF Thresholds."
)
parser.add_argument("--train", "-t", action="store_true", help="Flag to run training")
parser.add_argument("--eval", "-e", action="store_true", help="Flag to run evaluation")
parser.add_argument(
    "--model-path", type=str, default="model.pth", help="Path to save/load the model"
)
args = parser.parse_args()

# Load configuration
config_filepath = "data/config.yaml"
with open(config_filepath, "r") as file:
    config = yaml.safe_load(file)

model_params = config["model"]
training_params = config["training"]

# Initialize data loaders
train_loader, test_loader = get_data_loaders(
    training_params["batch_size"], training_params["data_path"]
)

device = torch.device("cpu")

# Grid search parameters
prune_options = [True, False]
threshold_options = [1.0, 2.0]

# Record results
results = []

for prune in prune_options:
    for lif_threshold in threshold_options:
        print(f"\nRunning with prune={prune}, LIF threshold={lif_threshold}")

        # Update model_params with the current LIF threshold
        model_params["threshold"] = lif_threshold

        # Initialize model with the specified LIF threshold
        net = Net(
            model_params["num_inputs"],
            model_params["num_hidden"],
            model_params["num_outputs"],
            model_params["num_steps"],
            model_params["beta"],
            threshold=model_params["threshold"],
        ).to(device)

        if args.train:
            # Train the model
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
            model_save_path = (
                f"models/model_prune_{prune}_threshold_{lif_threshold}.pth"
            )
            torch.save(net.state_dict(), model_save_path)

            # Optionally, plot the loss curves
            plt.figure(figsize=(10, 5))
            plt.plot(loss_hist)
            plt.plot(test_loss_hist)
            plt.title(f"Loss Curves (Prune: {prune}, Threshold: {lif_threshold})")
            plt.legend(["Train Loss", "Test Loss"])
            plt.savefig(f"results/loss_prune_{prune}_threshold_{lif_threshold}.png")
            plt.close()

        if args.eval:
            # Load the trained model
            model_load_path = (
                f"models/model_prune_{prune}_threshold_{lif_threshold}.pth"
            )
            net.load_state_dict(torch.load(model_load_path))

            if prune:
                # Prune weights
                prune_threshold = 0.1  # Adjust as needed
                prune_weights(net, prune_threshold)

            # Evaluate the model
            total, correct = test(net, test_loader, device, config)
            accuracy = correct / total * 100
            print(f"Accuracy: {accuracy:.2f}%")

            # Save results
            results.append(
                {"Pruned": prune, "LIF Threshold": lif_threshold, "Accuracy": accuracy}
            )

# Save all results to a file
if args.eval and results:
    import csv

    with open("results/grid_search_results.csv", mode="w", newline="") as results_file:
        writer = csv.DictWriter(
            results_file, fieldnames=["Pruned", "LIF Threshold", "Accuracy"]
        )
        writer.writeheader()
        for result in results:
            writer.writerow(result)
