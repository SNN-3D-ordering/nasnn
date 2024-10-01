import torch
import matplotlib.pyplot as plt


class MockNeuronCluster:
    def __init__(self, num_neurons):
        # Initialize coordinates randomly
        self.coordinates = torch.rand(
            num_neurons, 2
        )  # Assuming 2D coordinates for simplicity

    def cluster_neurons_simple(self, spk, factor=0.01):
        """Looks at all neurons that fired in spk (can be a timestep or whole batch) and moves them closer to each other."""
        # get sum of spk over batch dimension
        spk_sum = torch.sum(spk, dim=0)
        avg_spk_sum = torch.mean(spk_sum)

        # Plot initial coordinates
        self.plot_coordinates(spk_sum=spk_sum, title="Initial Coordinates")

        # get indices of neurons that fired
        fired_indices = torch.nonzero(spk_sum).squeeze()

        if fired_indices.numel() == 0:
            return  # skip clustering

        fired_coordinates = self.coordinates[fired_indices]


        # move each neuron that fired (factor*spk_sum[neuron coordinates]) closer to the average of all neurons that fired
        self.coordinates[fired_indices] += (
            factor
            * spk_sum[fired_indices].reshape(-1, 1)
            * (torch.mean(fired_coordinates, dim=0) - fired_coordinates)
        )

        # debugging
        print((
            factor
            * spk_sum[fired_indices].reshape(-1, 1)
            * (torch.mean(fired_coordinates, dim=0) - fired_coordinates)
        ))

        # move all neurons away from the center by the average amount that they moved towards the center
        center = torch.mean(self.coordinates, dim=0)
        self.coordinates += factor * (center - self.coordinates) * avg_spk_sum

        # Plot coordinates after clustering
        self.plot_coordinates(spk_sum=spk_sum, title="Coordinates After Clustering")

    def plot_coordinates(self, spk_sum=None, title="Neuron Coordinates"):
        x_coords = self.coordinates[:, 0].numpy()
        y_coords = self.coordinates[:, 1].numpy()

        if spk_sum is not None:
            colors = spk_sum.numpy()
        else:
            colors = "b"  # Default color if spk_sum is not provided

        plt.scatter(x_coords, y_coords, c=colors, cmap="viridis", s=100, edgecolor="k")
        plt.colorbar(label="Spike Sum")
        plt.title(title)
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.show()

num_neurons = 1000
mock_cluster = MockNeuronCluster(num_neurons)

# Generate a sample spike tensor (batch size 128, num_neurons 10) with Gaussian distribution
spk = torch.randn(128, num_neurons).float()

mock_cluster.cluster_neurons_simple(spk, factor=0.05)
