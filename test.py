# TODO: Heatmaps/ wann feuert welches neuron?
# TODO: Array übertragen in json
# TODO: weight matrices ==layer_connections in NetworkRepresentation
import torch
from utils import visualize_neuron_positions
import numpy as np

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
    # for debugging: visualize the neuron positions (2D continuous space)
    visualize_neuron_positions(net)
    return 

def record(net, test_loader, device, max_steps=None):
    # TODO build function that records spikes for N steps
    pass

def cluster_simple(net, test_loader, device, max_steps=None):
    # TODO build function that performs the naive clustering algorithm
    with torch.no_grad():
        net.eval()
        net.set_cluster_simple(True)

        #print("Clustering for ", np.max(max_steps, len(enumerate(test_loader))), " steps.")
        
        for step, (data, targets) in enumerate(test_loader):
            if max_steps is not None and step >= max_steps:
                break

            data = data.to(device)
            targets = targets.to(device)

            _, _ = net(data.view(data.size(0), -1))

        visualize_neuron_positions(net)
        return None
