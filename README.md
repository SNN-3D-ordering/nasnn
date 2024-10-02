# nasnn
_A neuron-arranging spiking neural network, an alternative neuron-arranging algorithm, and a spike distance metric for the whole network. Part of a project for the seminar "Diving into Neuromorphic Computing", summer semester of 2024, University of Osnabrück_

## Installation
To install and run this project, follow these steps:

1. Clone this repository and navigate to the project directory.
2. Run the following command to create the conda environment:
    ```
    conda env create -f environment.yml
    ```
3. Activate the newly created environment.


## Usage
Current usage instructions (subject to change)

1. Run the following command to train a model
    ```
    python main.py --train
    ```
2. Run the following command to test the model and do the clustering
    ```
    python main.py --eval
    ```

Use both flags to do both in the same run.


