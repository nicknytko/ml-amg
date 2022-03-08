import torch
import torch.linalg as tla
import torch.nn as nn
import torch_geometric.nn as tgnn
import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import sys
import sys
import matplotlib.pyplot as plt
import argparse

sys.path.append('../')
import ns.model.data

def parse_bool_str(v):
    v = v.lower()
    if v == 't' or v == 'true':
        return True
    else:
        return False

parser = argparse.ArgumentParser(description='Demo of the aggregate-picking network')
parser.add_argument('system', type=str, help='Grid file to visualize')
args = parser.parse_args()

grid = ns.model.data.Grid.load(args.system)
grid.plot()
plt.gca().set_aspect('equal')
plt.show(block=True)

print('DoF', grid.A.shape[0])
