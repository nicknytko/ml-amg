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

import common

parser = argparse.ArgumentParser(description='Demo of the trained ML-AMG network on a single example')
parser.add_argument('system', type=str, help='Problem selection to run demo on')
parser.add_argument('--model', type=str, help='Model file to evaluate')
parser.add_argument('--n', type=int, default=None, help='Size of the system.  In 2d, this determines the legnth in one dimension')
parser.add_argument('--alpha', type=float, default=None, help='Coarsening ratio for aggregation')
parser.add_argument('--spiderplot', type=common.parse_bool_str, default=True, help='Enable spider plot')
parser.add_argument('--strength-measure', default='olson', choices=common.strength_measure_funcs.keys())
parser.add_argument('--seed', type=int, default=None)
args = parser.parse_args()

N = args.n
alpha = args.alpha
neumann_solve = False

if os.path.exists(args.system):
    if alpha is None:
        alpha = .1
    grid = ns.model.data.Grid.load(args.system)
    A = grid.A
    n = A.shape[0]
    np.random.seed(0)
    C = common.strength_measure_funcs[args.strength_measure](A)
    Agg, Agg_roots = pyamg.aggregation.lloyd_aggregation(C, ratio=alpha, distance='same')
    figure_size = (10,10)
else:
    raise RuntimeError(f'Unknown system {args.system}')

np.random.seed()

model = ns.model.agg_interp.FullAggNet(64, num_conv=2, iterations=4)
model.load_state_dict(torch.load(args.model))
model.eval()

r = np.random.RandomState(seed=args.seed)
input_seed = np.zeros(n)
input_seed[r.choice(n, size=int(np.ceil(alpha*n)), replace=False)] = 1.

with torch.no_grad():
    agg_T, P_T, bf_weights, cluster_centers, node_scores = model.forward(A, alpha)
    intermediate = model.forward_intermediate_topk(A, alpha)

P_ML = ns.lib.sparse.torch_to_scipy(P_T)
P_Lloyd = ns.lib.multigrid.smoothed_aggregation_jacobi(A, Agg)

b = np.zeros(n)
x = np.random.RandomState(0).randn(n)
x /= la.norm(x, 2)

conv_ML = ns.lib.multigrid.amg_2_v(A, P_ML, b, x, res_tol=1e-6, singular=neumann_solve, jacobi_weight=2./3.)[1]
conv_Lloyd = ns.lib.multigrid.amg_2_v(A, P_Lloyd, b, x, res_tol=1e-6, singular=neumann_solve, jacobi_weight=2./3.)[1]

print('ML', conv_ML)
print('Lloyd', conv_Lloyd)

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
plt.savefig('agg_pass.pdf')

plt.show(block=True)
