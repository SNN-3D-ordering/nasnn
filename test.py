# TODO: Heatmaps/ wann feuert welches neuron?
# TODO: Array übertragen in json
# TODO: weight matrices ==layer_connections in NetworkRepresentation
import torch
from utils import visualize_neuron_positions_color, visualize_heatmaps
from network_representation import NetworkRepresentation
import numpy as np

import torch
import matplotlib.pyplot as plt


def test(net, test_loader, device, config, pruned=False, max_steps=None):
    # Evaluate model and record heatmaps
    total = 0
    correct = 0

    with torch.no_grad():
        net.eval()
        net.record_heatmap = True
        for data, targets in test_loader:

            data = data.to(device)
            targets = targets.to(device)

            test_spk, _ = net(data.view(data.size(0), -1))

            _, predicted = test_spk.sum(dim=0).max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            net.record_heatmap = False

    layers, weight_matrices, heatmaps = net.export_model_structure()
    network_representation = NetworkRepresentation(layers, weight_matrices, heatmaps)

    visualize_neuron_positions_color(net.lif1)
    # visualize_heatmaps(heatmaps)

    if pruned:
        export_filepath = config["filepaths"]["pruned_network_representation_filepath"]
    else:
        export_filepath = config["filepaths"]["network_representation_filepath"]

    print("Recording heatmaps to ", export_filepath)
    network_representation.export_representation(export_filepath)

    return total, correct


import torch


def record(net, test_loader, device, config, pruned=False, record_batches=None):
    # Record spike times for 100 batches
    with torch.no_grad():
        net.eval()
        net.record_spike_times = True
        batch_count = 0

        for data, targets in test_loader:
            if record_batches is not None and batch_count >= record_batches:
                break  # Stop recording after record_batches

            data = data.to(device)
            targets = targets.to(device)

            _, _ = net(data.view(data.size(0), -1))

            batch_count += 1

        print("Recorded spike times for ", batch_count, " batches.")
        net.record_heatmap = False

    layers, weight_matrices, heatmaps = net.export_model_structure()
    network_representation = NetworkRepresentation(layers, weight_matrices, heatmaps)

    activations = (
        net.get_activations()
    )  # Call the method to get activations #TODO get these within export_activations

    if pruned:
        export_filepath = config["filepaths"]["pruned_network_activations_filepath"]
    else:
        export_filepath = config["filepaths"]["network_activations_filepath"]
    print("Recording activations to ", export_filepath)
    network_representation.export_activations(activations, export_filepath)


def cluster_simple(net, test_loader, device, config, pruned=False, max_steps=None):
    # TODO let this load a specified nr of samples up to test_loader max amount (print message)
    with torch.no_grad():
        net.eval()
        net.set_cluster_simple(True)

        for step, (data, targets) in enumerate(test_loader):
            if max_steps is not None and step >= max_steps:
                break

            data = data.to(device)
            targets = targets.to(device)

            _, _ = net(data.view(data.size(0), -1))

        if net.heatmaps is not None:
            visualize_neuron_positions_color(net.lif1, spk_sum=net.heatmaps[1])
        else:
            visualize_neuron_positions_color(net.lif1)

        layers, weight_matrices, heatmaps = net.export_model_structure()
        network_representation = NetworkRepresentation(
            layers, weight_matrices, heatmaps
        )
        if pruned:
            export_filepath = config["filepaths"][
                "simple_clustered_pruned_network_representation_filepath"
            ]
        else:
            export_filepath = config["filepaths"][
                "simple_clustered_network_representation_filepath"
            ]
        print(f"Clustering done. Writing to {export_filepath}...")
        network_representation.export_representation(export_filepath)

        return None
