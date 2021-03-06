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
import ns.lib.graph

import common

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
parser.add_argument('--strength-measure', default='olson', choices=common.strength_measure_funcs.keys())
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
    Agg, Agg_roots, Agg_seeds = ns.lib.graph.lloyd_aggregation(C, ratio=alpha, distance='same')

    _, dumb_centers = ns.lib.graph.modified_bellman_ford(ns.lib.sparse.scipy_to_torch(C), torch.Tensor(Agg_seeds).long())
    Agg_dumb = ns.lib.sparse.torch_to_scipy(ns.lib.graph.nearest_center_to_agg(torch.Tensor(Agg_seeds).long(), dumb_centers))

    figure_size = (10,10)
elif args.system == '2d_unstructured':
    if alpha is None:
        alpha = .1
    grid = ns.model.data.Grid.random_2d_unstructured(N)
    A = grid.A
    n = A.shape[0]
    print(n)
    np.random.seed(0)
    C = common.strength_measure_funcs[args.strength_measure](A)
    Agg, Agg_roots, Agg_seeds = ns.lib.graph.lloyd_aggregation(C, ratio=alpha, distance='same')

    _, dumb_centers = ns.lib.graph.modified_bellman_ford(ns.lib.sparse.scipy_to_torch(C), torch.Tensor(Agg_seeds).long())
    Agg_dumb = ns.lib.sparse.torch_to_scipy(ns.lib.graph.nearest_center_to_agg(torch.Tensor(Agg_seeds).long(), dumb_centers))

    figure_size = (10,10)
elif args.system == '1d_dirichlet':
    figure_size = (10,3)
    if alpha is None:
        alpha = 1. / 3.
    if N is None:
        N = 9
    n = N
    grid = ns.model.data.Grid.structured_1d_poisson_dirichlet(n)
    A = grid.A
    n_aggs = int(n * alpha)
    agg_size = n // n_aggs

    Agg = np.zeros((n, n_aggs))
    Agg_roots = (np.arange(n_aggs) * agg_size) + (agg_size // 2)
    for agg in range(n_aggs):
        Agg[agg_size*agg:agg_size*(agg+1), agg] = 1.
    Agg = sp.csr_matrix(Agg)
elif args.system == '1d_neumann':
    neumann_solve = True
    figure_size = (10,3)

    if alpha is None:
        alpha = 1. / 3.
    if N is None:
        N = 9
    n = N
    grid = ns.model.data.Grid.structured_1d_poisson_neumann(n)
    A = grid.A
    n_aggs = int(n * alpha)
    agg_size = n // n_aggs

    Agg = np.zeros((n, n_aggs))
    Agg_roots = (np.arange(n_aggs) * agg_size) + (agg_size // 2)
    for agg in range(n_aggs):
        Agg[agg_size*agg:agg_size*(agg+1), agg] = 1.
    Agg = sp.csr_matrix(Agg)
elif args.system == '2d_isotropic':
    figure_size = (10,10)
    if alpha is None:
        alpha = .1
    if N is None:
        N = 8
    n = N**2
    grid = ns.model.data.Grid.structured_2d_poisson_dirichlet(N, N)
    A = grid.A

    np.random.seed(0)
    Agg, Agg_roots = pyamg.aggregation.lloyd_aggregation(A, ratio=alpha)
elif args.system == '2d_anisotropic':
    figure_size = (10,10)
    if alpha is None:
        alpha = .1
    if N is None:
        N = 8
    n = N**2
    grid = ns.model.data.Grid.structured_2d_poisson_dirichlet(N, N, epsilon=0.001, theta=np.pi/6)
    A = grid.A

    np.random.seed(0)
    Agg, Agg_roots = pyamg.aggregation.lloyd_aggregation(A, ratio=alpha)
else:
    print(f'Unknown system {args.system}')
    exit(1)

np.random.seed()

device = 'cpu'

A_T = ns.lib.sparse.to_torch_sparse(A).to(device)

# Set up Jacobi smoother
Dinv = sp.diags([1.0 / A.diagonal()], [0])
omega = (4. / 3.) / np.abs(spla.eigs(Dinv @ A, k=1, return_eigenvectors=False)).item()
smoother = (sp.eye(n) - omega*Dinv@A)
P_SA = smoother @ Agg
P_Dumb = smoother @ Agg_dumb

# Create our graph representation of A
A_Graph = ns.model.data.graph_from_matrix_basic(A)

def compute_agg_and_p(model):
    C = common.strength_measure_funcs[args.strength_measure](A)
    with torch.no_grad():
        agg, P, bf_weights, cluster_centers, node_scores = model.forward(A, alpha)
        agg_T = agg

    return agg_T, P, bf_weights, cluster_centers, node_scores

def loss_fcn(A, P_T):
    P = ns.lib.sparse_tensor.to_scipy(P_T)
    b = np.zeros(A.shape[1])
    x = np.random.randn(A.shape[1])
    x /= la.norm(x, 2)

    ret = ns.lib.multigrid.amg_2_v(A, P, b, x, res_tol=1e-10, singular=neumann_solve, jacobi_weight=omega)
    return ret[1]


def node_values(A, P):
    b = np.zeros(A.shape[1])
    x = np.random.randn(A.shape[1])
    x /= la.norm(x, 2)

    ret = ns.lib.multigrid.amg_2_v(A, P, b, x, res_tol=1e-10, max_iter=10, singular=neumann_solve, jacobi_weight=omega)
    return np.abs(ret[0])


def plot_grid(agg, P, bf_weights, cluster_centers, node_scores):
    graph = grid.networkx
    graph.remove_edges_from(list(nx.selfloop_edges(graph)))
    positions = {}
    for node in graph.nodes:
        positions[node] = grid.x[node]

    edge_values = np.zeros(len(graph.edges))
    for i, edge in enumerate(graph.edges):
        edge_values[i] = bf_weights[edge]

    if isinstance(node_scores, torch.Tensor):
        node_scores = node_scores.numpy()

    if not isinstance(P, sp.spmatrix):
        P = ns.lib.sparse_tensor.to_scipy(P)

    grid.plot_agg(agg, alpha=0.1, edgecolor='0.2')
    if args.spiderplot:
        grid.plot_spider_agg(agg, P)
    nx.drawing.nx_pylab.draw_networkx(graph, ax=plt.gca(), pos=positions, arrows=False, with_labels=False, node_size=50, width=0.1, node_color=node_scores)
    plt.plot(grid.x[cluster_centers, 0], grid.x[cluster_centers, 1], 'r*', markersize=6)
    plt.gca().set_aspect('equal')

plt.figure(figsize=figure_size)
plot_grid(Agg, P_SA, C, Agg_roots, node_values(A, P_SA))
plt.title(f'Baseline Lloyd + Jacobi, conv={loss_fcn(A, ns.lib.sparse.to_torch_sparse(P_SA)):.4f}')

plt.figure(figsize=figure_size)
plot_grid(Agg_dumb, P_Dumb, C, Agg_seeds, node_values(A, P_Dumb))
plt.title(f'Lloyd seeds + Jacobi, conv={loss_fcn(A, ns.lib.sparse.to_torch_sparse(P_Dumb)):.4f}')

model = ns.model.agg_interp.FullAggNet(64, num_conv=2, iterations=4)
model.load_state_dict(torch.load(args.model))
model.eval()

agg_T, P, bf_weights, cluster_centers, node_scores = compute_agg_and_p(model)
plt.figure(figsize=figure_size)
plot_grid(ns.lib.sparse_tensor.to_scipy(agg_T), P, bf_weights, cluster_centers, node_values(A, ns.lib.sparse.torch_to_scipy(P)))
conv = loss_fcn(A, P)

if 'theta' in grid.extra and 'epsilon' in grid.extra:
    plot_title = f'ML AMG, conv={conv:.4f}, theta={grid.extra["theta"]/np.pi:.2f}??, epsilon={grid.extra["epsilon"]:.3e}'
    plt.title(plot_title)

    theta = grid.extra['theta']
    epsilon = grid.extra['epsilon']
    c, s = np.cos(theta), np.sin(theta)
    Q = np.array([
        [c, -s],
        [s, c]
    ])
    A = np.diag([1., epsilon])
    D = Q@A@Q.T
    v1 = (Q[:,0]) / (1+epsilon)
    v2 = (Q[:,1] * epsilon) / (1+epsilon)
    va = v1+v2

    inset_axes = inset_axes(plt.gca(), width=1, height=1, bbox_transform=plt.gca().transAxes, bbox_to_anchor=(0.0, 0.0), loc=3)

    inset_axes.arrow(0, 0, v1[0], v1[1], head_width=.1, head_length=.1)
    inset_axes.arrow(0, 0, v2[0], v2[1], head_width=.1, head_length=.1, color='red')
    inset_axes.set_xlim(left=-1, right=1)
    inset_axes.set_ylim(bottom=-1, top=1)
else:
    plot_title = f'ML AMG, conv={conv:.4f}'
    plt.title(plot_title)
plt.show()
print('Convergence', conv)
