import torch
import torch.linalg as tla
import torch.nn as nn
import torch_geometric.nn as tgnn
import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import sys
import os
import pyamg
import matplotlib.pyplot as plt
import pygad
import pygad.torchga
import argparse
import time

sys.path.append('../')
import ns.model.agg_interp
import ns.model.data
import ns.lib.sparse
import ns.lib.sparse_tensor
import ns.lib.multigrid

parser = argparse.ArgumentParser(description='Demo of the aggregate-picking network')
parser.add_argument('system', type=str, help='Problem in data folder to evaluate')
args = parser.parse_args()

ds = ns.model.data.Grid.load_dir(args.system)
ds_size = list(map(lambda grid: grid.A.shape[0], ds))

plt.figure()
plt.hist(ds_size)
plt.title('Frequency of Problem Sizes')
plt.xlabel('Degrees of Freedom')
plt.ylabel('Number of Occurrences')
plt.show(block=True)
