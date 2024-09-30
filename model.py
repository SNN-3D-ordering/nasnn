import snntorch as snn
import torch
import torch.nn as nn
from custom_neurons import CustomLeaky
from network_representation import NetworkRepresentation
from utils import generate_square_grid_ascending


class Net(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, num_steps, beta):
        super().__init__()
        self.num_steps = num_steps
        self.current_step = 0
        self.testing = False # TODO remove
        self.record_heatmap = False
        self.heatmaps = []
        self.activations = [] # TODO rename heatmaps/activations to something better
        self.cluster_simple = False
        self.record_spike_times = False

        self.input_grid = generate_square_grid_ascending(num_inputs)
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = CustomLeaky(beta=beta, input_size=num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = CustomLeaky(beta=beta, input_size=num_outputs)

    def forward(self, x):
        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record heatmaps
        if self.record_heatmap:
            heatmap0 = torch.zeros((self.fc1.in_features,), device=x.device)
            self.heatmaps.append(heatmap0)
            heatmap1 = torch.zeros((self.fc1.out_features,), device=x.device)
            self.heatmaps.append(heatmap1)
            heatmap2 = torch.zeros((self.fc2.out_features,), device=x.device)
            self.heatmaps.append(heatmap2)

        if self.record_spike_times:
            activations0 = []
            activations1 = []
            activations2 = []

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for _ in range(self.num_steps):
            # Pass current step and weights for CustomLeaky layers
            self.current_step += 1
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1, self.current_step, self.fc1.weight.t())

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2, self.current_step, self.fc2.weight.t())

            if self.record_heatmap:
                heatmap0 += x.sum(dim=0)
                heatmap1 += spk1.sum(dim=0)
                heatmap2 += spk2.sum(dim=0)

            if self.record_spike_times:
                activations0 += x.sum(dim=0)
                activations1 += spk1.sum(dim=0)
                activations2 += spk2.sum(dim=0)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        if self.record_heatmap:
            self.heatmaps[0] += heatmap0
            self.heatmaps[1] += heatmap1
            self.heatmaps[2] += heatmap2
    
        if self.record_spike_times:
            self.activations.append([activations0, activations1, activations2])

        return torch.stack(spk2_rec), torch.stack(mem2_rec)
    
    def export_model_structure(self):
        """Returns all spiking layers as 2d grids and weight matrices of fully connected layers.

        Returns:
            list, list: list of 2d grids, list of weight matrices
        """
        layers = [self.input_grid]
        weight_matrices = []
        heatmaps = self.heatmaps

        for module in self.modules():
            if isinstance(module, CustomLeaky):
                layers.append(module.return_2d_grid())
            elif isinstance(module, nn.Linear):
                weight_matrices.append(module.weight.data)

        return layers, weight_matrices, heatmaps
    
    def get_activations(self):
        """Returns activations of all spiking layers.

        Returns:
            list: list of activations
        """
        return self.activations

    def set_cluster_simple(self, cluster_simple):
        """Sets the cluster_simple attribute of all CustomLeaky layers."""
        for module in self.modules():
            if isinstance(module, CustomLeaky):
                module.set_cluster_simple(cluster_simple)


        