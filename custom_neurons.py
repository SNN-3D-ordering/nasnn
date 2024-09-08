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
        self.positions_history[current_step] = self.coordinates.clone().detach().cpu().numpy()

    def cluster_neurons(self, spk):
        # TODO extend, this is only k-means for now
        # get all neurons (indices) that fired
        fired_indices = torch.nonzero(spk, as_tuple=True)
        if len(fired_indices[0]) == 0:
            return  # skip clustering
    
        # get neuron indices (assuming spk has shape [batch_size, num_neurons]), then coordinates
        neuron_indices = fired_indices[1]
        fired_coordinates = self.coordinates[neuron_indices]
    
        # assess nr of unique points to avoid k-means error
        unique_fired_coordinates = np.unique(fired_coordinates.cpu().numpy(), axis=0)
        num_unique_points = len(unique_fired_coordinates)
    
        # k-means clustering on fired neurons
        n_clusters = min(5, num_unique_points)  # Ensure n_clusters <= number of unique points
        if n_clusters > 1:  # Only perform clustering if there is more than one unique point
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(fired_coordinates)
    
            # update coordinates of fired neurons
            for i, idx in enumerate(neuron_indices):
                self.coordinates[idx] = torch.tensor(
                    kmeans.cluster_centers_[kmeans.labels_[i]]
                )

    def forward(self, input, mem, current_step):
        spk, mem = super().forward(input, mem)
        self.update_firing_times(spk, current_step)
        self.cluster_neurons(spk)
        return spk, mem

    def export_positions_history(self, file_path):
        # convert NumPy arrays to lists for JSON serialization
        serializable_history = {step: positions.tolist() for step, positions in self.positions_history.items()}
        
        # write to JSON
        with open(file_path, 'w') as json_file:
            json.dump(serializable_history, json_file, indent=4)