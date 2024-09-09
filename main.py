import torch
import yaml
from model import Net
from data import get_data_loaders
from train import train_model
from test import test
import matplotlib.pyplot as plt
import argparse

# Argument parser for command line flags
parser = argparse.ArgumentParser(description="Train or evaluate the model.")
parser.add_argument("--train", "-t", action="store_true", help="Flag to run training")
parser.add_argument("--eval", "-e", action="store_true", help="Flag to run evaluation")
parser.add_argument(
    "--model-path", type=str, default="model.pth", help="Path to save/load the model"
)
args = parser.parse_args()

# Load configuration
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

model_params = config["model"]
training_params = config["training"]

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

# Use CPU for now (this is because the indices and tensors are not on the same device)
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
    torch.save(net.state_dict(), args.model_path)

    # Plot the loss curves
    fig = plt.figure(facecolor="w", figsize=(10, 5))
    plt.plot(loss_hist)
    plt.plot(test_loss_hist)
    plt.title("Loss Curves")
    plt.legend(["Train Loss", "Test Loss"])
    plt.show()

if args.eval:
    # Load the trained model
    net.load_state_dict(torch.load(args.model_path))

    # Evaluate model
    total, correct = test(net, test_loader, device)
    print(f"Accuracy: {correct/total*100:.2f}%")

if not args.train and not args.eval:
    print("Please specify either --train/-t or --eval/-e flag.")
