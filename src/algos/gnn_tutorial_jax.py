## Standard libraries
import os
import json
import math
import numpy as np
import time

## Imports for plotting
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg', 'pdf') # For export
from matplotlib.colors import to_rgb
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns
sns.reset_orig()
sns.set()

## Progress bar
from tqdm.notebook import tqdm

## To run JAX on TPU in Google Colab, uncomment the two lines below
# import jax.tools.colab_tpu
# jax.tools.colab_tpu.setup_tpu()

## JAX
import jax
import jax.numpy as jnp
from jax import random
# Seeding for random operations
main_rng = random.PRNGKey(42)

## Flax (NN in JAX)
import flax
import optax
from flax import linen as nn


## PyTorch
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms

# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = "../../data"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "../../saved_models/tutorial7_jax"

# print("Device:", jax.devices()[0])

# import urllib.request
# from urllib.error import HTTPError
# # Github URL where saved models are stored for this tutorial
# base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/JAX/tutorial7/"
# # Files to download
# pretrained_files = []

# # Create checkpoint path if it doesn't exist yet
# os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# # For each file, check whether it already exists. If not, try downloading it.
# for file_name in pretrained_files:
#     file_path = os.path.join(CHECKPOINT_PATH, file_name)
#     if "/" in file_name:
#         os.makedirs(file_path.rsplit("/",1)[0], exist_ok=True)
#     if not os.path.isfile(file_path):
#         file_url = base_url + file_name
#         print(f"Downloading {file_url}...")
#         try:
#             urllib.request.urlretrieve(file_url, file_path)
#         except HTTPError as e:
#             print("Something went wrong. Please contact the author with the full output including the following error:\n", e)


# class GCNLayer(nn.Module):
#     c_out : int  # Output feature size
    
#     @nn.compact
#     def __call__(self, node_feats, adj_matrix):
#         """
#         Inputs:
#             node_feats - Array with node features of shape [batch_size, num_nodes, c_in]
#             adj_matrix - Batch of adjacency matrices of the graph. If there is an edge from i to j, adj_matrix[b,i,j]=1 else 0.
#                          Supports directed edges by non-symmetric matrices. Assumes to already have added the identity connections. 
#                          Shape: [batch_size, num_nodes, num_nodes]
#         """
#         # Num neighbours = number of incoming edges
#         num_neighbours = adj_matrix.sum(axis=-1, keepdims=True)
#         node_feats = nn.Dense(features=self.c_out, name='projection')(node_feats)
#         node_feats = jax.lax.batch_matmul(adj_matrix, node_feats)
#         node_feats = node_feats / num_neighbours
#         return node_feats

# node_feats = jnp.arange(8, dtype=jnp.float32).reshape((1, 4, 2))
# adj_matrix = jnp.array([[[1, 1, 0, 0],
#                             [1, 1, 1, 1],
#                             [0, 1, 1, 1],
#                             [0, 1, 1, 1]]]).astype(jnp.float32)

# print("Node features:\n", node_feats)
# print("\nAdjacency matrix:\n", adj_matrix)


def smooth_l1_loss(x, y, delta=1.0):
    diff = jnp.abs(x - y)
    smooth_l1 = jnp.where(diff < delta, 0.5 * diff**2, delta * (diff - 0.5 * delta))
    return smooth_l1.mean()

predictions = jnp.array([0.8, 1.7, 5.5])
targets = jnp.array([1.0, 1.5, 4.0])
loss = smooth_l1_loss(predictions, targets)
print("Smooth L1 Loss JAX:", loss)

def smooth_l1_loss_mine(x, y, beta=1.0):
    diff = jnp.abs(x - y)
    smooth_l1 = jnp.where(diff < beta, (0.5 * diff**2)/beta, (diff - 0.5 * beta))
    return smooth_l1.mean()

predictions = jnp.array([0.8, 1.7, 5.5])
targets = jnp.array([1.0, 1.5, 4.0])
loss = smooth_l1_loss_mine(predictions, targets)
print("Smooth L1 Loss JAX-myversion:", loss)

predictions = torch.tensor([0.8, 1.7, 5.5])
targets = torch.tensor([1.0, 1.5, 4.0])
loss_torch = F.smooth_l1_loss(predictions, targets)
print("Smooth L1 Loss:", loss_torch)