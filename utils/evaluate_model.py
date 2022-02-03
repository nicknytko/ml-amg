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
        alpha = 1. / 9.
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
        alpha = 1. / 9.
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

# Create PyTorch tensor versions of everything
smoother_T = ns.lib.sparse.to_torch_sparse(smoother).to(device)
Agg_T = ns.lib.sparse.to_torch_sparse(Agg).to(device)
Agg_roots_T = torch.Tensor(Agg_roots)
A_Graph = ns.model.data.graph_from_matrix_basic(A)

def compute_agg_and_p(model):
    C = strength_of_measure_funcs[args.strength_measure](A)
    with torch.no_grad():
        agg_T, P, bf_weights, cluster_centers, node_scores = model.forward(A, alpha)
        agg_sp = ns.lib.sparse_tensor.to_scipy(agg_T)

    return agg_T, P, bf_weights, cluster_centers, node_scores

def loss_fcn(A, P_T):
    P = ns.lib.sparse_tensor.to_scipy(P_T)
    b = np.zeros(A.shape[1])
    x = np.random.randn(A.shape[1])
    x /= la.norm(x, 2)

    ret = ns.lib.multigrid.amg_2_v(A, P, b, x, res_tol=1e-10, singular=neumann_solve, jacobi_weight=omega)
    return ret[1]

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
    node_scores = np.log10(node_scores + 1)

    if not isinstance(P, sp.spmatrix):
        P = ns.lib.sparse_tensor.to_scipy(P)

    grid.plot_agg(agg, alpha=0.1, edgecolor='0.2')
    if args.spiderplot:
        grid.plot_spider_agg(agg, P)
    nx.drawing.nx_pylab.draw_networkx(graph, ax=plt.gca(), pos=positions, arrows=False, with_labels=False, node_size=50, edge_color=edge_values, node_color=node_scores)
    plt.plot(grid.x[cluster_centers, 0], grid.x[cluster_centers, 1], 'r*', markersize=6)
    plt.gca().set_aspect('equal')

plt.figure(figsize=figure_size)
plot_grid(Agg, P_SA, C, Agg_roots, torch.zeros(A.shape[0]))
plt.title(f'Baseline Lloyd + Jacobi, conv={loss_fcn(A, ns.lib.sparse.to_torch_sparse(P_SA)):.4f}')

model = ns.model.agg_interp.FullAggNet(64)
model.load_state_dict(torch.load(args.model))
model.eval()

agg_T, P, bf_weights, cluster_centers, node_scores = compute_agg_and_p(model)
plt.figure(figsize=figure_size)
plot_grid(ns.lib.sparse_tensor.to_scipy(agg_T), P, bf_weights, cluster_centers, node_scores)
conv = loss_fcn(A, P)

if 'theta' in grid.extra and 'epsilon' in grid.extra:
    plot_title = f'ML AMG, conv={conv:.4f}, theta={grid.extra["theta"]/np.pi:.2f}π, epsilon={grid.extra["epsilon"]:.3e}'
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
    # eigvals, eigvecs = la.eig(D)
    # eigvals /= la.norm(eigvals, np.inf)
    # v1 = eigvecs[:,0] * eigvals[0]
    # v2 = eigvecs[:,1] * eigvals[1]
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
