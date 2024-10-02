# an algorithm to reduce the total manhattan distance between neurons in the entire network

# 1) Do the following:
# Create a rank-network-representation like this:
# - load the network representation at network_representation_filepath (its a json file)
# - for the input layer (stored as first layer in the json):
#     - generate a rank-map from the heatmap (heatmap is a 2D array of the same size as the input layer, the rank-map is the heatmap with each value replaced by its rank in the heatmap)
#
# - for each layer (starting from the second layer):
#     - generate a binary weight matrix (a 2D array of the same size as the weight matrix, with 1s where the weight matrix is non-zero and 0s where it is zero)
#           - alternatively: 1s for the top x% of the weight matrix and 0s for the rest
#     - generate a heat-map from the (heatmap of previous layer) * (previous weight matrix) (element-wise multiplication)
#     - turn this new heat-map into a rank-map
#
# Use the rank-network-representation to reduce the total manhattan distance between neurons in the entire network:
# - move neurons of identical ranks in the rank-network-representation closer together
# - for each layer (starting from the second layer):

# 2) Alternatively:
# Generate a similarity-score for each pair of layers (layer1, layer2) by comparing the rank-maps of the heatmaps of the layers
# - each neuron gets its similarty score by comparing its rank in the heatmap of layer1 with its rank in the heatmap of layer2
# - the similarity score is 1 - the absolute difference between the two ranks
# the similarity score of the layers is the average of the similarity scores of all neurons
# - do simulated annealing to minimize the similarity scores between layers

# 3) Alternatively alternatively:
# Do the same as above, but use a kernel instead of neuron pairs
# - the similarity score of the layers is the average similarity score of all kernel pairs
# - the similarity score of a kernel pair is the average similarity score of all neuron pairs in the kernel pair
#  (maybe we sort in the kernel)

# 4) Alternatively alternatively alternatively:
# Use the similarity score of the layers to generate a similarity matrix
# - the similarity matrix is a 2D array with the similarity score of each pair of layers
# - use the similarity matrix as a distance matrix to do hierarchical clustering
# - use the hierarchical clustering to generate a dendrogram
# - cut the dendrogram at the desired number of clusters
# - move neurons of the same cluster closer together
