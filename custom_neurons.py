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

    def cluster_neurons_simple(self, spk):
        """Looks at all neurons that fired in spk (can be a timestep or whole batch) and moves them closer to each other."""
        # get indices of neurons that fired
        fired_indices = torch.nonzero(spk, as_tuple=True)

        # TODO spk shape is always [batch size, num_neurons of next layer]
        # fired_indices[0].shape == fired_indices[1].shape
        if len(fired_indices[0]) == 0:
            return  # skip clustering

        fired_coordinates = self.coordinates[fired_indices[1]]

        # move all neurons away from the center a bit
        center = torch.mean(fired_coordinates, dim=0)
        self.coordinates += 0.01 * (self.coordinates - center)

        # move all neurons that fired closer to each other
        center = torch.mean(fired_coordinates, dim=0)
        self.coordinates[fired_indices[1]] += 0.01 * (fired_coordinates - center)

    def cluster_neurons_simple_stepwise(self, spk):
        """go stepwise (for batch_size steps) over the neurons that fired, each time, do one step of cluster_neurons_simple"""
        fired_indices = torch.nonzero(spk, as_tuple=True)[1]
        if len(fired_indices) == 0:
            return

        # go stepwise (for len(fired_indices) steps) over the neurons that fired, each time, do one step of cluster_neurons_simple
        for i in range(len(fired_indices)):
            self.cluster_neurons_simple(spk[:, fired_indices[i]])

    def cluster_neurons_corr_matrix(self, spk):
        """Iterates through the batch, clustering neurons that fired together at each timestep."""
        # this moves those that fired together multiple times in a batch closer to each other than the ones that fired fewer times
        # should be a similar result to cluster_neurons_simple_stepwise, but might be more efficient

        # move all neuron coordinates outward from the center
        center = torch.mean(self.coordinates, dim=0)
        self.coordinates += 0.01 * (self.coordinates - center)

        # analyze (from the full batch of neurons) which ones tend to fire together
        correlation_matrix = torch.zeros(self.num_neurons, self.num_neurons)
        for i in range(spk.shape[0]):
            fired_indices = torch.nonzero(spk[i], as_tuple=True)[1]
            if len(fired_indices) == 0:
                return
            for j in range(len(fired_indices)):
                for k in range(j + 1, len(fired_indices)):
                    neuron1 = fired_indices[j]
                    neuron2 = fired_indices[k]
                    correlation_matrix[neuron1, neuron2] += 1
                    correlation_matrix[neuron2, neuron1] += 1

        # move the coordinates of the neurons that tend to fire together closer to each other. use the information from the correlation matrix
        for i in range(self.num_neurons):
            for j in range(i + 1, self.num_neurons):
                if correlation_matrix[i, j] > 0:
                    self.coordinates[i] += (
                        0.01
                        * correlation_matrix[i, j]
                        * (self.coordinates[j] - self.coordinates[i])
                    )
                    self.coordinates[j] += (
                        0.01
                        * correlation_matrix[i, j]
                        * (self.coordinates[i] - self.coordinates[j])
                    )

    def update_connections(self, spk, input_spk, weight_matrix):
        """Updates the connections between neurons based on the spikes at the current timestep."""
        # Get indices of neurons that fired in the current layer and the one before
        curr_indices = torch.nonzero(spk, as_tuple=True)[1]
        input_indices = torch.nonzero(input_spk, as_tuple=True)[1]
    
        # Print shapes for debugging
        print(f"input_spk shape: {input_spk.shape}")
        print(f"spk shape: {spk.shape}")
        print(f"weight_matrix shape: {weight_matrix.shape}")
    
        # Ensure that the weight matrix dimensions match the input and current layer sizes
        assert weight_matrix.shape[0] == input_spk.shape[1], "Weight matrix row size must match input layer size"
        assert weight_matrix.shape[1] == spk.shape[1], "Weight matrix column size must match current layer size"
    
        # Validate indices
        if curr_indices.max() >= weight_matrix.shape[1] or input_indices.max() >= weight_matrix.shape[0]:
            raise IndexError("Index out of bounds for weight matrix dimensions")
    
        # Create a mask for active connections (from non-zero weights)
        active_connections = weight_matrix[input_indices][:, curr_indices] != 0
    
        # Get the indices of active connections
        active_input_indices, active_curr_indices = torch.nonzero(active_connections, as_tuple=True)
    
        # Map the indices back to the original neuron indices
        active_input_indices = input_indices[active_input_indices]
        active_curr_indices = curr_indices[active_curr_indices]
    
        # Update connections based on the active input neurons
        for input_idx, curr_idx in zip(active_input_indices, active_curr_indices):
            connection = (input_idx.item(), curr_idx.item())
            if connection in self.connections:
                self.connections[connection] += 1
            else:
                self.connections[connection] = 1

                
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

    def forward(self, input, mem, current_step, weight_matrix): 
        spk, mem = super().forward(input, mem)
        # print input shape
        print(f"Input shape in fwd: {input.shape}")
        if not self.training:
            #self.update_firing_times(
            #    spk, current_step
            #)  # TODO check if we also want this for training / if we want it at all
            self.cluster_neurons_simple(spk)

            #self.update_connections(spk, input, weight_matrix)

        return spk, mem
