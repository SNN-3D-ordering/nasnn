import snntorch as snn
import torch
import numpy as np
import json
from utils import map_to_2d_grid_row_wise
from utils import convert_layer_size_to_grid_size



class CustomLeaky(snn.Leaky):
    def __init__(self, beta, input_size):
        super().__init__(beta=beta)
        self.num_neurons = input_size
        self.grid_dims = convert_layer_size_to_grid_size(input_size)
        self.coordinates = self.initialize_coordinates()
        self.firing_times = torch.zeros(input_size)
        self.positions_history = {}  # Dictionary to store positions at each timestep
        self.connections = {}  # Dictionary to store connections between neurons
        self.prev_spk = None
        self.cluster_simple = False

    def initialize_coordinates(self):
        # randomly initialized 2D coordinates for each neuron
        return torch.rand(self.num_neurons, 2)

    def update_firing_times(self, spk, current_step):
        """Updates the firing times of the neurons based on the current timestep and spikes."""
        # TODO remove if we use grid coords only
        # update firing times based on current timestep
        self.firing_times[spk.any(dim=0)] = current_step
        # store all positions at current timestep for later export
        self.positions_history[current_step] = (
            self.coordinates.clone().detach().cpu().numpy()
        )

    def cluster_neurons_simple(self, spk, factor=0.001, heat=0.1):
        """Looks at all neurons that fired in spk (can be a timestep or whole batch) and moves them closer to each other."""
        # get sum of spk over batch dimension
        spk_sum = torch.sum(spk, dim=0)
    
        # get indices of neurons that fired
        fired_indices = torch.nonzero(spk_sum).squeeze()
    
        if fired_indices.numel() == 0:
            return  # skip clustering
    
        fired_coordinates = self.coordinates[fired_indices]
    
        # move each neuron that fired (factor*spk_sum[neuron coordinates]) closer to the average of all neurons that fired
        inward_movement = (
            factor
            * spk_sum[fired_indices].reshape(-1, 1)
            * (torch.mean(fired_coordinates, dim=0) - fired_coordinates)
        )
    
        self.coordinates[fired_indices] += inward_movement

        total_inward_movement = torch.sum(inward_movement, dim=0)
    
        # Calculate the outward movement for all neurons proportionally
        center = torch.mean(self.coordinates, dim=0)
        distances_from_center = torch.norm(self.coordinates - center, dim=1, keepdim=True)
        outward_movement = (total_inward_movement / self.coordinates.size(0)) * (self.coordinates - center) / (distances_from_center + 1e-6)
    
        # Apply the outward movement
        self.coordinates += outward_movement
    
        # Add a small random perturbation to spread neurons out slightly
        if heat != 0:
            self.coordinates += torch.randn_like(self.coordinates) * heat


    # TODO build this:
    # def update_connections_cumulative(self, spk, input_spk, weight_matrix)
    # get: prev_spk * weight-matrix (check if not putting prev_spk through layer in reverse is better = prev
    # take spk, element wise multiply with prev
    # sum those all up, get a matrix of the previous layer weighted by which neurons are most relevant to the spiking ones of this layer
    # use that for the clustering later 
    
    def export_positions_history(self, file_path):
        """
        Exports the positions history to a JSON file.

        Args:
            file_path (str): path to store the JSON file.
        """
        # TODO potentially convert to work with grid coordinates instead of 2D coordinates
        # convert NumPy arrays to lists for JSON serialization
        serializable_history = {
            step: positions.tolist()
            for step, positions in self.positions_history.items()
        }

        # write to JSON
        with open(file_path, "w") as json_file:
            json.dump(serializable_history, json_file, indent=4)

    def return_2d_grid(self):
        """
        Maps the 2D coordinates to a 2D grid.

        Args:
            grid_size (tuple): Size of the grid (rows, cols).

        Returns:
            np.ndarray: 2d grid with neuron indices.
        """
        coordinates = self.coordinates.cpu().numpy()
        grid = map_to_2d_grid_row_wise(
            coordinates, self.grid_dims
        )  # TODO try out other mapping functions
        return grid
    
    def return_connections(self):
        """
        Returns the connections between of this layer and the previous layer. 

        Returns:
            dict: Dictionary with connections between neurons.
        """
        return self.connections
    
    def set_cluster_simple(self, cluster_simple):
        self.cluster_simple = cluster_simple

    def forward(self, input, mem, current_step, weight_matrix): 
        spk, mem = super().forward(input, mem)
        #if not self.training:
            #self.update_firing_times(
            #    spk, current_step
            #)  # TODO check if we also want this for training / if we want it at all
            #self.update_connections(spk, input, weight_matrix)

        return spk, mem
