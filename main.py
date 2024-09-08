import torch
from model import Net
from data import get_data_loaders
from train import train_model

# params
num_inputs = 28*28
num_hidden = 1000
num_outputs = 10

num_steps = 25
beta = 0.95

batch_size = 128
data_path = "/tmp/data/mnist"
dtype = torch.float

num_epochs = 1

train_loader, test_loader = get_data_loaders(batch_size, data_path)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

net = Net().to(device)

loss_hist, test_loss_hist = train_model(net, train_loader, test_loader, device, num_steps, num_epochs, batch_size, dtype)

# Plot Loss
import matplotlib.pyplot as plt

fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(loss_hist)
plt.plot(test_loss_hist)
plt.title("Loss Curves")
plt.legend(["Train Loss", "Test Loss"])
plt.xlabel("Iteration")
plt.ylabel("Loss")
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