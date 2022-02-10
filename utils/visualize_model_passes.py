import torch
import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import sys
import os
import pyamg
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pygad
import pygad.torchga
import argparse
import networkx as nx

sys.path.append('../')
import ns.model.agg_interp
import ns.model.data
import ns.lib.sparse
import ns.lib.sparse_tensor
import ns.lib.multigrid

def parse_bool_str(v):
    v = v.lower()
    if v == 't' or v == 'true':
        return True
    else:
        return False

parser = argparse.ArgumentParser(description='Demo of the trained ML-AMG network on a single example')
parser.add_argument('system', type=str, help='Problem selection to run demo on')
parser.add_argument('--model', type=str, help='Model file to evaluate')
parser.add_argument('--n', type=int, default=None, help='Size of the system.  In 2d, this determines the legnth in one dimension')
parser.add_argument('--alpha', type=float, default=None, help='Coarsening ratio for aggregation')
parser.add_argument('--spiderplot', type=parse_bool_str, default=True, help='Enable spider plot')
parser.add_argument('--strength-measure', default='abs', choices=['abs', 'evolution', 'invabs', 'unit', 'luke'])
args = parser.parse_args()

N = args.n
alpha = args.alpha
neumann_solve = False

strength_of_measure_funcs = {
    'abs': lambda A: abs(A),
    'evolution': lambda A: pyamg.strength.evolution_strength_of_connection(A) + sp.csr_matrix((np.ones_like(A.data), A.indices, A.indptr), A.shape) * 0.1,
    'luke': lambda A: pyamg.strength.evolution_strength_of_connection(A) + sp.csr_matrix((1./np.abs(A.data), A.indices, A.indptr), A.shape),
    'invabs': lambda A: sp.csr_matrix((1.0 / np.abs(A.data), A.indices, A.indptr), A.shape),
    'unit': lambda A: sp.csr_matrix((np.ones_like(A.data), A.indices, A.indptr), A.shape)
}

if os.path.exists(args.system):
    if alpha is None:
        alpha = 1. / 3.
    grid = ns.model.data.Grid.load(args.system)
    A = grid.A
    n = A.shape[0]
    np.random.seed(0)
    C = strength_of_measure_funcs[args.strength_measure](A)
    Agg, Agg_roots = pyamg.aggregation.lloyd_aggregation(C, ratio=alpha, distance='same')
    figure_size = (10,10)
else:
    raise RuntimeError(f'Unknown system {args.system}')

np.random.seed()

model = ns.model.agg_interp.FullAggNet(64)
model.load_state_dict(torch.load(args.model))
model.eval()

data_simple = ns.model.data.graph_from_matrix_basic(A)
with torch.no_grad():
    intermediate = model.AggNet.all_intermediate_topk(data_simple, int(np.ceil(alpha * A.shape[0])))

graph = grid.networkx
graph.remove_edges_from(list(nx.selfloop_edges(graph)))

positions = {}
for node in graph.nodes:
    positions[node] = grid.x[node]

fig, axes = plt.subplots(1, len(intermediate), figsize=(15,4))
plt.title('Values of aggregate centers vs binarization pass')

for i in range(len(intermediate)):
    ax = axes[i]
    cluster_centers = (intermediate[i] == 1.0)
    nx.drawing.nx_pylab.draw_networkx(graph, ax=ax, pos=positions, arrows=False, with_labels=False, node_size=20)
    ax.plot(grid.x[cluster_centers, 0], grid.x[cluster_centers, 1], 'k*', markersize=6)
    ax.set_aspect('equal')
    ax.set_title(f'Pass {i+1}')

plt.show(block=True)
