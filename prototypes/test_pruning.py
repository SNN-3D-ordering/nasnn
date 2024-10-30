import torch
import torch.nn as nn


def prune_weights(network, threshold):
    """
    Set all weights in the network that are below the threshold to zero.

    Args:
        network (nn.Module): The neural network to prune.
        threshold (float): The threshold below which weights are set to zero.
    """
    for name, param in network.named_parameters():
        if "weight" in name:
            with torch.no_grad():
                # Create a mask of weights below the threshold
                mask = param.abs() < threshold
                # Set weights below the threshold to zero
                param[mask] = 0


# Define a simple network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(4, 2)


# Initialize the network
net = SimpleNet()

# Manually set weights for testing
with torch.no_grad():
    net.fc.weight = nn.Parameter(
        torch.tensor([[0.05, -0.1, 0.2, -0.3], [0.4, -0.5, 0.6, -0.7]])
    )

print("Weights before pruning:")
print(net.fc.weight)

# Prune weights with threshold 0.3
prune_weights(net, threshold=0.3)

print("\nWeights after pruning with threshold 0.3:")
print(net.fc.weight)
