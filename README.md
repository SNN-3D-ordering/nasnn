# SNN-3D-Ordering

## Overview

**SNN-3D-Ordering** is a project focused on optimizing the spatial arrangement of Spiking Neural Networks (SNNs) on 3D neuromorphic hardware. This repository contains code for training SNNs, clustering neurons to optimize routing, and analyzing communication efficiency.

## Table of Contents

- [SNN-3D-Ordering](#snn-3d-ordering)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Features](#features)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Setup](#setup)
  - [Usage](#usage)
    - [Training](#training)
    - [Clustering](#clustering)
  - [License](#license)

## Introduction

Neuromorphic hardware architectures mimic the complex, three-dimensional connectivity of biological neural networks, presenting opportunities for efficient deployment of SNNs. This repository provides two clustering algorithms to optimize neuron placement:

1. **Activation-Based Clustering (ABC)**: Groups neurons with high firing activity together within each layer, reducing spatial communication costs.
2. **Rank-Alignment (RA)**: Aligns neurons based on rank similarity in firing activity across layers, minimizing inter-layer signal distances.

These clustering techniques aim to improve network efficiency by reducing signal travel distances.

## Features

- **SNN Training**: Implements a basic SNN in Python using SNNtorch, trained on the MNIST dataset.
- **Clustering Algorithms**:
  - **Activation-Based Clustering**: Clusters neurons by firing patterns.
  - **Rank-Alignment**: Aligns neurons by similarity of activity ranks across layers.

## Installation

### Prerequisites

- Python 3.9+
- [SNNtorch](https://snntorch.readthedocs.io/en/latest/)

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/SNN-3D-ordering/nasnn
   cd nasnn
   ```

2. **Install required packages**:
   ```bash
   conda env create -f environment.yml
   ```

## Usage

### Training

Train an SNN on the MNIST dataset using `training.py`.

- **Train the model**:
  ```bash
  python main.py -t
  ```

### Clustering

Use `clustering.py` to optimize neuron placement.

- **Normale**:
  ```bash
  python main.py -e
  ```

- **Pruned**:
  ```bash
  python main.py -ep
  ```

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.