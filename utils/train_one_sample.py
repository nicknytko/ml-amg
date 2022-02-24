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

import common

def parse_bool_str(v):
    v = v.lower()
    if v == 't' or v == 'true':
        return True
    else:
        return False

parser = argparse.ArgumentParser(description='Demo of the aggregate-picking network')
parser.add_argument('system', type=str, help='Problem selection to run demo on')
parser.add_argument('--n', type=int, default=None, help='Size of the system.  In 2d, this determines the legnth in one dimension')
parser.add_argument('--max-generations', type=int, default=500, help='Maximum number of training generations')
parser.add_argument('--initial-population-size', type=int, default=50, help='Initial population size')
parser.add_argument('--alpha', type=float, default=None, help='Coarsening ratio for aggregation')
args = parser.parse_args()

N = args.n
neumann_solve = False
alpha = args.alpha

if os.path.exists(args.system):
    if alpha is None:
        alpha = 1. / 3.
    grid = ns.model.data.Grid.load(args.system)
    A = grid.A
    n = A.shape[0]
    np.random.seed(0)
    C = common.strength_measure_funcs['olson'](A)
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
agg_v = np.zeros(A.shape[1])
agg_v[Agg_roots] = 1.

# Model
model = ns.model.agg_interp.AggOnlyNet(64)
model_folds = [
    'AggNet.layers.0',
    'AggNet.layers.1',
    'AggNet.layers.2',
    'AggNet.layers.3',
    'AggNet.feature_map'
]

def print_mat_rows(P, round_digits=5):
    if isinstance(P, torch.Tensor) and P.is_sparse:
        P = P.to_dense().numpy()
    else:
        P = np.array(P.todense())
    P = P / la.norm(P, ord=np.inf, axis=0)

    for row in range(P.shape[0]):
        p = np.array(P[row]).flatten()
        s_ = '[ '
        for p_i in p:
            s = str(p_i)
            if len(s) > round_digits+2:
                s = s[:round_digits+2]
            else:
                s = s + (' ' * (round_digits+2 - len(s)))
            s_ += s + ' '
        s_ += ']'
        print(s_)

def compute_agg_and_p(weights, random):
    model.load_state_dict(ns.ga.torch.model_weights_as_dict(model, weights))
    model.eval()

    n = A.shape[1]
    x = np.zeros(n)
    x[random.choice(n, size=int(np.ceil(alpha*n)), replace=False)] = 1.

    with torch.no_grad():
        agg_sp, P, bf_weights, cluster_centers, node_scores = model(A, alpha, x=x, C_in=C)
        agg_T = ns.lib.sparse.scipy_to_torch(agg_sp)

    return agg_T, P, bf_weights, cluster_centers, node_scores

def loss_fcn(A, P_T):
    P = ns.lib.sparse_tensor.to_scipy(P_T)
    b = np.zeros(A.shape[1])
    r = np.random.RandomState(seed=0)
    x = r.randn(A.shape[1])
    x /= la.norm(x, 2)

    ret = ns.lib.multigrid.amg_2_v(A, P, b, x, res_tol=1e-10, singular=neumann_solve, jacobi_weight=omega)
    return ret[1]

def fitness(generation, weights, weights_idx):
    r = np.random.RandomState(seed=generation)
    agg, P, bf_weights, cluster_centers, node_scores = compute_agg_and_p(weights, r)
    loss = loss_fcn(A, P)

    if np.isnan(loss.item()) or loss.item() == 1.:
        return 0
    else:
        return 1.0 / loss

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

    plt.gcf().set_size_inches(figure_size[0], figure_size[1])
    nx.drawing.nx_pylab.draw_networkx(graph, ax=plt.gca(), pos=positions, arrows=False, with_labels=False, node_size=60, edge_color=edge_values, node_color=node_scores)
    grid.plot_agg(agg, alpha=0.1, edgecolor='0.2')
    grid.plot_spider_agg(agg, P)
    plt.plot(grid.x[cluster_centers, 0], grid.x[cluster_centers, 1], 'y*', markersize=10)
    plt.gca().set_aspect('equal')

def display_progress(ga_instance):
    weights = ga_instance.best_solution()
    gen = ga_instance.num_generation

    r = np.random.RandomState(seed=gen)
    agg, P, bf_weights, cluster_centers, node_scores = compute_agg_and_p(weights[0], r)
    agg = ns.lib.sparse_tensor.to_scipy(agg)
    loss = loss_fcn(A, P)

    print(f'Generation = {gen}')
    print(f'Fitness    = {weights[1]}')
    print(f'Loss       = {loss}')

    plt.clf()
    plot_grid(agg, P, bf_weights, cluster_centers, node_scores)
    plt.title(f'Generation {gen}, MLAMG, conv={loss:.4f}')

    plt.show()
    plt.pause(1)

    model.load_state_dict(pygad.torchga.model_weights_as_dict(model, weights[0]))
    model.eval()

if __name__ == '__main__':
    baseline_conv = loss_fcn(A, ns.lib.sparse.to_torch_sparse(P_SA))
    print(f'Baseline convergence {baseline_conv:.4f}')
    plt.ion()
    plt.figure(figsize=figure_size)
    plot_grid(Agg, P_SA, A, Agg_roots, torch.zeros(A.shape[0]))
    plt.title(f'Baseline, conv={baseline_conv:.4f}')
    plt.show()
    plt.pause(1)

    population = ns.ga.torch.TorchGA(model=model, num_solutions=args.initial_population_size, model_fold_names=model_folds)
    initial_population = population.population_weights

    perturb_val = 1
    num_workers = 4
    ga_instance = ns.ga.parga.ParallelGA(initial_population=initial_population,
                                         fitness_func=fitness,
                                         crossover_probability=0.5,
                                         mutation_probability=0.4,
                                         mutation_min_perturb=-perturb_val,
                                         mutation_max_perturb=perturb_val,
                                         steady_state_top_use=2./3.,
                                         steady_state_bottom_discard=1./4.,
                                         num_workers=num_workers,
                                         model_folds=population.folds)
    ga_instance.start_workers()
    display_progress(ga_instance)

    for i in range(args.max_generations):
        ga_instance.stochastic_iteration()
        display_progress(ga_instance)

    ga_instance.finish_workers()
