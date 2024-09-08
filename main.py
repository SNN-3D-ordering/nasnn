import torch
import yaml
from model import Net
from data import get_data_loaders
from train import train_model
import matplotlib.pyplot as plt

# Load configuration
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

model_params = config["model"]
training_params = config["training"]

# Set device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Initialize data loaders
train_loader, test_loader = get_data_loaders(
    training_params["batch_size"], training_params["data_path"]
)

# Initialize model
net = Net(
    model_params["num_inputs"],
    model_params["num_hidden"],
    model_params["num_outputs"],
    model_params["num_steps"],
    model_params["beta"],
).to(device)

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

# Plot the loss curves
fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(loss_hist)
plt.plot(test_loss_hist)
plt.title("Loss Curves")
plt.legend(["Train Loss", "Test Loss"])
plt.show()

# Test accuracy over all 10,000 samples
total = 0
correct = 0

with torch.no_grad():
    net.eval()
    for data, targets in test_loader:
        data = data.to(device)
        targets = targets.to(device)

        test_spk, _ = net(data.view(data.size(0), -1))

        _, predicted = test_spk.sum(dim=0).max(1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
