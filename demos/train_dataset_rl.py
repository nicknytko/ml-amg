import torch
import torch.linalg as tla
import torch.nn as nn
import torch_geometric.nn as tgnn
from torch.utils import tensorboard
import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import sys
import os
import pyamg
import matplotlib.pyplot as plt
import argparse
import networkx as nx

sys.path.append('../')
import ns.model.agg_interp
import ns.model.data
import ns.lib.sparse
import ns.lib.sparse_tensor
import ns.lib.multigrid
import ns.lib.graph
import ns.ga.parga
import ns.ga.torch
import utils.common as common


parser = argparse.ArgumentParser(description='Demo of the aggregate-picking network')
parser.add_argument('system', type=str, help='Problem in data folder to train on')
parser.add_argument('--max-generations', type=int, default=500, help='Maximum number of training generations')
parser.add_argument('--initial-population-size', type=int, default=20, help='Initial population size')
parser.add_argument('--alpha', type=float, default=0.1, help='Coarsening ratio for aggregation')
parser.add_argument('--workers', type=int, default=3, help='Number of workers to use for parallel GA training')
parser.add_argument('--start-generation', type=int, default=0, help='Initial generation (used for resuming training)')
parser.add_argument('--start-model', type=str, default=None, help='Initial generation (used for resuming training)')
parser.add_argument('--strength-measure', default='abs', choices=common.strength_measure_funcs.keys())
args = parser.parse_args()


class AggLayer(nn.Module):
    def __init__(self, dim):
        super(AggLayer, self).__init__()
        ncs = []
        ecs = []
        fcs = []
        norms = []

        # Input -> Hidden
        ncs.append(tgnn.TAGConv(1, dim))
        # ecs.append(EdgeConvModel(dim, 1, 1, dim))
        fcs.append(
            nn.Sequential(
                nn.Linear(dim, dim), nn.ReLU(),
                nn.Linear(dim, 1), nn.ReLU()
            )
        )

        self.ncs = nn.ModuleList(ncs)
        self.fcs = nn.ModuleList(fcs)


    def forward(self, x, edge_index, edge_attr):
        if len(x.shape) == 1:
            x = torch.unsqueeze(x, 1)
        n = x.shape[0]

        for i in range(1):
            x = nn.functional.relu(self.ncs[i](x, edge_index, edge_attr))
            x = self.fcs[i](x)

        return x


# Load problem
grid = ns.model.data.Grid.load(args.system)
A = grid.A
n = A.shape[0]
alpha = args.alpha
target_k = int(np.ceil(n*alpha))
np.random.seed(0)
C = common.strength_measure_funcs['olson'](A)
C_T = ns.lib.sparse.scipy_to_torch(C)
Agg, Agg_roots = pyamg.aggregation.lloyd_aggregation(C, ratio=alpha, distance='same')
data = ns.model.data.graph_from_matrix_basic(A)
model = AggLayer(32)


def plot_grid(node_scores):
    graph = grid.networkx
    graph.remove_edges_from(list(nx.selfloop_edges(graph)))
    positions = {}
    for node in graph.nodes:
        positions[node] = grid.x[node]

    if isinstance(node_scores, torch.Tensor):
        node_scores = node_scores.numpy()

    plt.gcf().set_size_inches((8,8))
    nx.drawing.nx_pylab.draw_networkx(graph, ax=plt.gca(), pos=positions, arrows=False, with_labels=False, node_size=60, node_color=node_scores, vmin=0, vmax=1)
    plt.gca().set_aspect('equal')



def loss(c):
    num_c = np.sum(c)
    if num_c != target_k:
        return (target_k - np.sum(c))**2

    distance, nearest_center = ns.lib.graph.modified_bellman_ford(C_T, torch.Tensor(c))
    agg_T = ns.lib.graph.nearest_center_to_agg(c, nearest_center)
    agg = ns.lib.sparse_tensor.to_scipy(agg_T)
    P = ns.lib.multigrid.smoothed_aggregation_jacobi(A, agg)

    x = np.random.RandomState(0).randn(A.shape[1])
    x /= la.norm(x, 2)
    res = ns.lib.multigrid.amg_2_v(A, P, b, x, res_tol=1e-6, singular=neumann_solve, jacobi_weight=omega)[1]

    if np.isnan(res):
        ell = 0
    else:
        ell = res

    return ell


i = 0
ite = 10
lr = 1e-1

# plt.ion()
# plt.figure(figsize=(8,8))
# plot_grid(torch.zeros(A.shape[0]))
# plt.pause(0.5)

while True:
    g = []
    for i, param in enumerate(model.parameters()):
        g.append(torch.zeros_like(param))

    for i in range(ite):
        q = torch.sigmoid(model(data.x, data.edge_index, data.edge_attr))
        q_numpy = q.detach().cpu().numpy().flatten()

        c_out = []
        for j in range(n):
            q_i = q_numpy[j]
            c_out.append(np.random.choice(2, size=1, replace=True, p=[1-q_i, q_i]))
        c = np.array(c_out).flatten()
        c_T = torch.Tensor(c)
        p = torch.sum(torch.log(c_T*q + (1-c_T)*(1-q)))
        grad = torch.autograd.grad(p, model.parameters())
        ell = loss(c)
        print(p)
        for j in range(len(g)):
            g[j] += ell * grad[j] / ite

    with torch.no_grad():
        for i, param in enumerate(model.parameters()):
            param -= lr * g[i]
