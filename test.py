import torch
from utils import visualize_neuron_positions

def test(net, test_loader, device, max_steps=None):
    total = 0
    correct = 0

    with torch.no_grad():
        net.eval()
        for step, (data, targets) in enumerate(test_loader):
            if max_steps is not None and step >= max_steps:
                break

            data = data.to(device)
            targets = targets.to(device)

            test_spk, _ = net(data.view(data.size(0), -1))

            _, predicted = test_spk.sum(dim=0).max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    visualize_neuron_positions(net)
    return total, correct