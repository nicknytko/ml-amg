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

def parse_bool_str(v):
    v = v.lower()
    if v == 't' or v == 'true':
        return True
    else:
        return False

parser = argparse.ArgumentParser(description='Demo of the aggregate-picking network')
parser.add_argument('system', choices=['1d_neumann', '1d_dirichlet', '2d_isotropic', '2d_anisotropic'], help='Problem selection to run demo on')
parser.add_argument('--n', type=int, default=None, help='Size of the system.  In 2d, this determines the legnth in one dimension')
parser.add_argument('--max-generations', type=int, default=500, help='Maximum number of training generations')
parser.add_argument('--initial-population-size', type=int, default=50, help='Initial population size')
parser.add_argument('--ml-aggregator', type=parse_bool_str, default=True, help='Whether to use the ML network for aggregation (true) or use some baseline (false)')
parser.add_argument('--ml-interpolator', type=parse_bool_str, default=True, help='Whether to use the ML network for interpolation (true) or use a Jacobi smoother (false)')
parser.add_argument('--alpha', type=float, default=None, help='Coarsening ratio for aggregation')
args = parser.parse_args()

N = args.n
neumann_solve = False
ml_aggregator = args.ml_aggregator
ml_interpolator = args.ml_interpolator

if ml_aggregator and ml_interpolator:
    suffix = 'full'
    title = 'ML aggregates and interpolation'
elif ml_aggregator:
    suffix = 'agg'
    title = 'ML Aggregates, SA'
else:
    suffix = 'interp'
    title = 'Lloyd, ML interpolation'

fig_directory = f'figures/{args.system}_figures_{suffix}'
alpha = args.alpha

if not os.path.exists(fig_directory):
    os.mkdir(fig_directory)

if args.system == '1d_dirichlet':
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
print(' -- Problem setup --')
print('A\n', A.todense())

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

def compute_agg_and_p(weights):
    model = ns.model.agg_interp.FullAggNet(64, use_aggnet=ml_aggregator, use_pnet=ml_interpolator)
    model.load_state_dict(pygad.torchga.model_weights_as_dict(model, weights))
    model.eval()

    with torch.no_grad():
        if not ml_aggregator:
            agg_T = Agg_T
            P = model.forward_fixed_agg(A, Agg)
            bf_weights = A_T
            cluster_centers = Agg_roots_T.int()
            node_scores = torch.zeros(A.shape[0])
        else:
            agg_T, P, bf_weights, cluster_centers, node_scores = model.forward(A, alpha)
            agg_sp = ns.lib.sparse_tensor.to_scipy(agg_T)
            if not ml_interpolator:
                P = ns.lib.sparse.to_torch_sparse(smoother @ agg_sp)

    return agg_T, P, bf_weights, cluster_centers, node_scores

def loss_fcn(A, P_T):
    P = ns.lib.sparse_tensor.to_scipy(P_T)
    b = np.zeros(A.shape[1])
    x = np.random.randn(A.shape[1])
    x /= la.norm(x, 2)

    ret = ns.lib.multigrid.amg_2_v(A, P, b, x, res_tol=1e-10, singular=neumann_solve, jacobi_weight=omega)
    return ret[1]

def fitness(weights, weights_idx):
    agg, P, bf_weights, cluster_centers, node_scores = compute_agg_and_p(weights)
    loss = loss_fcn(A, P)

    if np.isnan(loss.item()) or loss.item() == 1.:
        return 0
    else:
        return 1.0 / loss

def plot_grid(agg, bf_weights, cluster_centers, node_scores):
    graph = grid.networkx
    positions = {}
    for node in graph.nodes:
        positions[node] = grid.x[node]

    edge_values = np.zeros(len(graph.edges))
    for i, edge in enumerate(graph.edges):
        edge_values[i] = bf_weights[edge]

    if isinstance(node_scores, torch.Tensor):
        node_scores = node_scores.numpy()
    node_scores = np.log10(node_scores + 1)

    grid.plot_agg(agg)
    nx.drawing.nx_pylab.draw_networkx(graph, ax=plt.gca(), pos=positions, arrows=False, with_labels=False, node_size=100, edge_color=edge_values, node_color=node_scores)
    plt.plot(grid.x[cluster_centers, 0], grid.x[cluster_centers, 1], 'r*', markersize=6)
    plt.gca().set_aspect('equal')

plt.ion()
plt.figure(figsize=figure_size)
plot_grid(Agg, A, Agg_roots, torch.zeros(A.shape[0]))
plt.title(f'Baseline, conv={loss_fcn(A, ns.lib.sparse.to_torch_sparse(P_SA)):.4f}')
plt.savefig(f'{fig_directory}/baseline.pdf')
plt.pause(0.1)

def display_progress(ga_instance):
    weights = ga_instance.best_solution()
    gen = ga_instance.generations_completed

    agg, P, bf_weights, cluster_centers, node_scores = compute_agg_and_p(weights[0])
    agg = ns.lib.sparse_tensor.to_scipy(agg)

    print(f'Generation = {gen}')
    print(f'Fitness    = {weights[1]}')
    print(f'Loss       = {1.0/weights[1]}')

    plt.clf()
    plot_grid(agg, bf_weights, cluster_centers, node_scores)
    plt.title(f'Generation {gen}, {title}, conv={1.0/weights[1]:.4f}')
    plt.pause(0.1)
    plt.savefig(f'{fig_directory}/{gen}_agg.pdf')

    model = ns.model.agg_interp.FullAggNet(64, use_aggnet=ml_aggregator, use_pnet=ml_interpolator)
    model.load_state_dict(pygad.torchga.model_weights_as_dict(model, weights[0]))
    model.eval()
    torch.save(model.state_dict(), f'models/{args.system}_{suffix}')

if __name__ == '__main__':
    model = ns.model.agg_interp.FullAggNet(64, use_aggnet=ml_aggregator, use_pnet=ml_interpolator)
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
