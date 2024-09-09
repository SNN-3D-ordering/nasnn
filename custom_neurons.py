import snntorch as snn
import torch
import numpy as np
from sklearn.cluster import KMeans
import json


class CustomLeaky(snn.Leaky):
    def __init__(self, beta, input_size):
        super().__init__(beta=beta)
        self.num_neurons = input_size
        self.coordinates = self.initialize_coordinates()
        self.firing_times = torch.zeros(input_size)
        self.positions_history = {}  # Dictionary to store positions at each timestep

    def initialize_coordinates(self):
        # randomly initialized 2D coordinates for each neuron
        return torch.rand(self.num_neurons, 2)

    def calculate_distance(self, coord1, coord2):
        # euclidean distance
        return torch.sqrt(torch.sum((coord1 - coord2) ** 2))

    def update_firing_times(self, spk, current_step):
        # update firing times based on current timestep
        self.firing_times[spk.any(dim=0)] = current_step
        # store all positions at current timestep for later export
        self.positions_history[current_step] = (
            self.coordinates.clone().detach().cpu().numpy()
        )

    def cluster_neurons_simple(self, spk):
        # TODO extend, this will only move about the neurons a bit, but should be fine to test whether anything happens at all

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

    def cluster_neurons_batchwise(self, spk):
        # 0. move all neuron coordinates outward from the center
        # 1. remove any neurons that did not fire
        # 2. analyze (from the full batch of neurons) which ones tend to fire together

        # 3. move the coordinates of the neurons that tend to fire together closer to each other

        # move all neuron coordinates outward from the center
        center = torch.mean(self.coordinates, dim=0)
        self.coordinates += 0.01 * (self.coordinates - center)

        # remove any neurons that did not fire (TODO: check if this is necessary)

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

    def cluster_neurons_stepwise(self, spk):
        # 1. remove any neurons that did not fire
        # 2. go stepwise (for batch_size steps) over the neurons that fired, each time, do one step of cluster_neurons_simple
        # 3. repeat for the next batch_size neurons that fired

        # remove any neurons that did not fire (TODO: check if this is necessary)
        fired_indices = torch.nonzero(spk, as_tuple=True)[1]
        if len(fired_indices) == 0:
            return
        
        # go stepwise (for len(fired_indices) steps) over the neurons that fired, each time, do one step of cluster_neurons_simple
        for i in range(len(fired_indices)):
            self.cluster_neurons_simple(spk[:, fired_indices[i]])


    def export_positions_history(self, file_path):
        # convert NumPy arrays to lists for JSON serialization
        serializable_history = {
            step: positions.tolist()
            for step, positions in self.positions_history.items()
        }

        # write to JSON
        with open(file_path, "w") as json_file:
            json.dump(serializable_history, json_file, indent=4)

    def forward(self, input, mem, current_step):
        spk, mem = super().forward(input, mem)

        # TODO fix train/test mode behavior (currently eval mode always runs)
        if self.training:
            # training mode behavior (normal forward pass)
            spk, mem = super().forward(input, mem)
        else:
            # eval mode behavior (custom clustering)
            current_step = input.shape[0]  # TODO this is wrong (always 128)
            self.update_firing_times(spk, current_step)
            self.cluster_neurons_simple(spk)
        return spk, mem
