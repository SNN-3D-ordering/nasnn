# TODO: Heatmaps/ wann feuert welches neuron?
# TODO: Array übertragen in json
# TODO: weight matrices ==layer_connections in NetworkRepresentation

import torch
import matplotlib.pyplot as plt


def test(net, test_loader, device):
    total = 0
    correct = 0

    with torch.no_grad():
        net.eval()
        net.set_record_heatmap(True)  # Enable heatmap recording
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)

            test_spk, _ = net(data.view(data.size(0), -1))

            _, predicted = test_spk.sum(dim=0).max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()


            net.set_record_heatmap(False)

    visualize_neuron_positions(net)

    return total, correct


def visualize_neuron_positions(net):
    coordinates = net.lif1.coordinates.detach().numpy()
    plt.scatter(coordinates[:, 0], coordinates[:, 1])
    plt.show()
