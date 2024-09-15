import snntorch as snn
import torch
import torch.nn as nn
from custom_neurons import CustomLeaky
from network_representation import NetworkRepresentation


class Net(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, num_steps, beta):
        super().__init__()
        self.num_steps = num_steps
        self.current_step = 0

        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = CustomLeaky(beta=beta, input_size=num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = CustomLeaky(beta=beta, input_size=num_outputs)

    def forward(self, x):
        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
    
        # Record the final layer
        spk2_rec = []
        mem2_rec = []
    
        for _ in range(self.num_steps):
            # Pass current step and weights for CustomLeaky layers
            self.current_step += 1
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1, self.current_step, self.fc1.weight.t())
            
            # Update connections using the spikes from the previous layer only in eval mode
            if not self.training:
                self.lif1.update_connections(spk1, x, self.fc1.weight.t())
            
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2, self.current_step, self.fc2.weight.t())
            
            # Update connections using the spikes from the previous layer only in eval mode
            if not self.training:
                self.lif2.update_connections(spk2, spk1, self.fc2.weight.t())
    
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)
    
        return torch.stack(spk2_rec), torch.stack(mem2_rec)

    def export_network_representation(self):
        # Instantiate NetworkRepresentation
        network_representation = NetworkRepresentation()

        # Get all CustomLeaky layers
        layers = []
        for module in self.modules():
            if isinstance(module, CustomLeaky):
                layers.append(module)

        # Export list of 2D grids and connections
        for layer_idx, layer in enumerate(layers):
            network_representation.add_layer(layer.return_2d_grid())

            connections = layer.return_connections()  # TODO is empty dict
            for neuron_i, neuron_j in connections:
                network_representation.add_connection(layer_idx, neuron_i, neuron_j)

        return network_representation
