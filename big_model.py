import scipy as sp
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
        
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.lif2 = CustomLeaky(beta=beta, input_size=num_hidden)
        
        self.fc3 = nn.Linear(num_hidden, num_hidden)
        self.lif3 = CustomLeaky(beta=beta, input_size=num_hidden)
        
        self.fc4 = nn.Linear(num_hidden, num_hidden)
        self.lif4 = CustomLeaky(beta=beta, input_size=num_hidden)
        
        self.fc5 = nn.Linear(num_hidden, num_hidden)
        self.lif5 = CustomLeaky(beta=beta, input_size=num_hidden)
        
        self.fc6 = nn.Linear(num_hidden, num_outputs)
        self.lif6 = CustomLeaky(beta=beta, input_size=num_outputs)

    def forward(self, x):
        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()
        mem6 = self.lif6.init_leaky()
    
        # Record heatmaps
        if self.record_heatmap:
            heatmap0 = torch.zeros((self.fc1.in_features,), device=x.device, dtype=torch.int64)
            self.heatmaps.append(heatmap0)
            heatmap1 = torch.zeros((self.fc1.out_features,), device=x.device, dtype=torch.int64)
            self.heatmaps.append(heatmap1)
            heatmap2 = torch.zeros((self.fc2.out_features,), device=x.device, dtype=torch.int64)
            self.heatmaps.append(heatmap2)
            heatmap3 = torch.zeros((self.fc3.out_features,), device=x.device, dtype=torch.int64)
            self.heatmaps.append(heatmap3)
            heatmap4 = torch.zeros((self.fc4.out_features,), device=x.device, dtype=torch.int64)
            self.heatmaps.append(heatmap4)
            heatmap5 = torch.zeros((self.fc5.out_features,), device=x.device, dtype=torch.int64)
            self.heatmaps.append(heatmap5)
            heatmap6 = torch.zeros((self.fc6.out_features,), device=x.device, dtype=torch.int64)
            self.heatmaps.append(heatmap6)      
              
        if self.record_spike_times:
            activations0 = []
            activations1 = []
            activations2 = []
            activations3 = []
            activations4 = []
            activations5 = []
            activations6 = []
    
        # Record the final layer
        spk6_rec = []
        mem6_rec = []
    
        for _ in range(self.num_steps):
            # Pass current step and weights for CustomLeaky layers
            self.current_step += 1
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1, self.current_step, self.fc1.weight.t())
    
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2, self.current_step, self.fc2.weight.t())
    
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3, self.current_step, self.fc3.weight.t())
    
            cur4 = self.fc4(spk3)
            spk4, mem4 = self.lif4(cur4, mem4, self.current_step, self.fc4.weight.t())
    
            cur5 = self.fc5(spk4)
            spk5, mem5 = self.lif5(cur5, mem5, self.current_step, self.fc5.weight.t())
    
            cur6 = self.fc6(spk5)
            spk6, mem6 = self.lif6(cur6, mem6, self.current_step, self.fc6.weight.t())
    
            if self.record_heatmap:
                heatmap0 += x.sum(dim=0).int()
                heatmap1 += spk1.sum(dim=0).int()
                heatmap2 += spk2.sum(dim=0).int()
                heatmap3 += spk3.sum(dim=0).int()
                heatmap4 += spk4.sum(dim=0).int()
                heatmap5 += spk5.sum(dim=0).int()
                heatmap6 += spk6.sum(dim=0).int()
    
            if self.record_spike_times:
                activations0 += x.sum(dim=0)
                activations1 += spk1.sum(dim=0)
                activations2 += spk2.sum(dim=0)
                activations3 += spk3.sum(dim=0)
                activations4 += spk4.sum(dim=0)
                activations5 += spk5.sum(dim=0)
                activations6 += spk6.sum(dim=0)
    
            spk6_rec.append(spk6)
            mem6_rec.append(mem6)
    
        if self.record_heatmap:
            self.heatmaps[0] += heatmap0
            self.heatmaps[1] += heatmap1
            self.heatmaps[2] += heatmap2
            self.heatmaps[3] += heatmap3
            self.heatmaps[4] += heatmap4
            self.heatmaps[5] += heatmap5
            self.heatmaps[6] += heatmap6

    
        if self.record_spike_times:
            self.activations.append([activations0, activations1, activations2, activations3, activations4, activations5, activations6])
    
        return torch.stack(spk6_rec), torch.stack(mem6_rec)
    
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
        