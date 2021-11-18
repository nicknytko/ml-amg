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
import ns.model.loss
import ns.model.data
import ns.lib.sparse
import ns.lib.sparse_tensor
import ns.lib.multigrid
import ns.lib.graph

parser = argparse.ArgumentParser(description='Demo of the aggregate-picking network')
parser.add_argument('--system', choices=['1d_neumann', '1d_dirichlet', '2d_isotropic', '2d_anisotropic'], required=True)
parser.add_argument('--n', type=int, default=None)
parser.add_argument('--max-generations', type=int, default=500)
parser.add_argument('--initial-population-size', type=int, default=50)
parser.add_argument('--ml-aggregator', type=bool, default=True)
parser.add_argument('--ml-interpolator', type=bool, default=True)
args = parser.parse_args()

N = args.n
neumann_solve = False
ml_aggregator = args.ml_aggregator
ml_interpolator = args.ml_interpolator
fig_directory = f'{args.system}_figures'

if not os.path.exists(fig_directory):
    os.mkdir(fig_directory)

if args.system == '1d_dirichlet':
    figure_size = (10,3)
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
    alpha = 1. / 9.
    if N is None:
        N = 8
    n = N**2
    grid = ns.model.data.Grid.structured_2d_poisson_dirichlet(N, N)
    A = grid.A
    Agg, Agg_roots = pyamg.aggregation.lloyd_aggregation(A, ratio=alpha)
elif args.system == '2d_anisotropic':
    figure_size = (10,10)
    alpha = 1. / 9.
    if N is None:
        N = 8
    n = N**2
    grid = ns.model.data.Grid.structured_2d_poisson_dirichlet(N, N, epsilon=0.001, theta=np.pi/6)
    A = grid.A
    Agg, Agg_roots = pyamg.aggregation.lloyd_aggregation(A, ratio=alpha)

device = 'cpu'

A_T = ns.lib.sparse.to_torch_sparse(A).to(device)
print(' -- Problem setup --')
print('A\n', A.todense())

omega = 2. / 3.
Dinv = sp.diags([1.0 / A.diagonal()], [0])
Agg_T = ns.lib.sparse.to_torch_sparse(Agg).to(device)
Agg_roots_T = torch.Tensor(Agg_roots)
smoother = (sp.eye(n) - omega*Dinv@A)
smoother_T = ns.lib.sparse.to_torch_sparse(smoother).to(device)
A_Graph = ns.model.data.graph_from_matrix_basic(A)
P_SA = smoother @ Agg

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
print_mat_rows(P_SA)

def compute_agg_and_p(weights):
    model = ns.model.agg_interp.FullAggNet(64)
    model.load_state_dict(pygad.torchga.model_weights_as_dict(model, weights))
    model.eval()

    with torch.no_grad():
        if not ml_aggregator:
            agg_T = Agg_T
            P = model.forward_fixed_agg(A, Agg)
            bf_weights = A_T
            cluster_centers = Agg_roots_T
        else:
            agg_T, P, bf_weights, cluster_centers = model.forward(A, alpha)
            agg_sp = ns.lib.sparse_tensor.to_scipy(agg_T)
            if not ml_interpolator:
                P = ns.lib.sparse.to_torch_sparse(smoother @ agg_sp)

    return agg_T, P, bf_weights, cluster_centers

def loss_fcn(A, P_T):
    P = ns.lib.sparse_tensor.to_scipy(P_T)
    b = np.zeros(A.shape[1])
    x = np.random.randn(A.shape[1])
    x /= la.norm(x, 2)

    ret = ns.lib.multigrid.amg_2_v(A, P, b, x, res_tol=1e-10, singular=neumann_solve)
    return ret[1]

def fitness(weights, weights_idx):
    model = ns.model.agg_interp.FullAggNet(64)
    model.load_state_dict(pygad.torchga.model_weights_as_dict(model, weights))
    model.eval()

    agg, P, bf_weights, cluster_centers = compute_agg_and_p(weights)
    loss = loss_fcn(A, P)

    if np.isnan(loss.item()):
        return 0
    else:
        return 1.0 / loss

def plot_grid(agg, bf_weights, cluster_centers):
    graph = grid.networkx
    positions = {}
    for node in graph.nodes:
        positions[node] = grid.x[node]

    edge_values = np.zeros(len(graph.edges))
    for i, edge in enumerate(graph.edges):
        edge_values[i] = bf_weights[edge]

    grid.plot_agg(agg)
    nx.drawing.nx_pylab.draw_networkx(graph, ax=plt.gca(), pos=positions, arrows=False, with_labels=False, node_size=100, edge_color=edge_values)
    plt.plot(grid.x[cluster_centers, 0], grid.x[cluster_centers, 1], 'y*', markersize=20)
    plt.gca().set_aspect('equal')

plt.ion()
plt.figure(figsize=figure_size)
plot_grid(Agg, A, Agg_roots)
plt.title(f'Baseline, conv={loss_fcn(A, ns.lib.sparse.to_torch_sparse(P_SA)):.4f}')
plt.savefig(f'{fig_directory}/baseline.pdf')
plt.pause(0.1)

def display_progress(ga_instance):
    weights = ga_instance.best_solution()
    gen = ga_instance.generations_completed

    agg, P, bf_weights, cluster_centers = compute_agg_and_p(weights[0])
    agg = ns.lib.sparse_tensor.to_scipy(agg)

    print(f'Generation = {gen}')
    print(f'Fitness    = {weights[1]}')
    print(f'Loss       = {1.0/weights[1]}')
    print('P')
    print_mat_rows(P)

    plt.clf()
    plot_grid(agg, bf_weights, cluster_centers)
    plt.title(f'Generation {gen}, conv={1.0/weights[1]:.4f}')
    plt.pause(0.1)
    plt.savefig(f'{fig_directory}/{gen}_agg.pdf')

if __name__ == '__main__':
    model = ns.model.agg_interp.FullAggNet(64)
    initial_population = pygad.torchga.TorchGA(model=model, num_solutions=args.initial_population_size).population_weights
    ga_instance = pygad.GA(num_generations=args.max_generations,
                           num_parents_mating=5,
                           initial_population=initial_population,
                           fitness_func=fitness,
                           on_generation=display_progress,
                           mutation_probability=0.3,
                           keep_parents=1,
                           num_parallel_workers=None)
    ga_instance.run()
