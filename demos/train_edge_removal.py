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
sys.path.append('../utils')
import ns.model.agg_interp
import ns.model.data
import ns.lib.sparse
import ns.lib.sparse_tensor
import ns.lib.multigrid
import ns.lib.graph
import ns.lib.aggplot
import ns.lib.disjoint_sets
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
parser.add_argument('--greedy', default=False, type=common.parse_bool_str)
parser.add_argument('--start-model', type=str, default=None, help='Initial generation (used for resuming training)')
args = parser.parse_args()

N = args.n
neumann_solve = False
alpha = args.alpha

greedy = args.greedy

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


class AggEdgeRemovalNet(nn.Module):
    def __init__(self, dim=64, num_conv=2, iterations=4):
        super(AggEdgeRemovalNet, self).__init__()

        self.PNet = ns.model.agg_interp.MPNN(dim, num_internal_conv=4, input_edge_features=2)
        self.EdgeNet = ns.model.agg_interp.MPNN(dim, num_internal_conv=4, input_edge_features=1)

    def forward(self, A, alpha):
        m, n = A.shape
        k = int(np.ceil(alpha * m))
        data_simple = ns.model.data.graph_from_matrix_basic(A)

        # Compute edge scores
        _, a_edges = self.EdgeNet(data_simple)
        edge_argsort = torch.argsort(a_edges.squeeze(), descending=True).squeeze()

        # remove edges to form disjoint sets
        num_sets = n
        ds = ns.lib.disjoint_sets.DisjointSets(n)
        i = 0
        while num_sets > k:
            ij = data_simple.edge_index[:, edge_argsort[i]]
            if ds.union(ij[0], ij[1]):
                num_sets -= 1
            i += 1
        sets = ds.get_disjoint_sets()

        # assemble aggregate assignment matrix
        rows = np.zeros(n, dtype=int)
        cols = np.zeros(n, dtype=int)
        i = 0
        for col, s in enumerate(sets):
            for row in s:
                rows[i] = row
                cols[i] = col
                i += 1
        agg = sp.coo_matrix((np.ones(n), (rows, cols)), shape=(m, k)).tocsr()
        agg_T = ns.lib.sparse.scipy_to_torch(agg)

        # Compute the smoother \hat{P}
        data = ns.model.data.graph_from_matrix(A, agg)
        _, p_edges = self.PNet(data)
        P_hat = torch.sparse_coo_tensor(data.edge_index, p_edges.squeeze(), (m, n)).coalesce()

        # Now, form P := \hat{P} Agg.
        P_T = torch.sparse.mm(P_hat, agg_T).coalesce()
        return agg_T, P_T, torch.sparse_coo_tensor(data_simple.edge_index, a_edges.squeeze(), (m, n)).coalesce()


# Model
model = AggEdgeRemovalNet(64, num_conv=2, iterations=4)
plot = None


def compute_agg_and_p(weights, random):
    model.load_state_dict(ns.ga.torch.model_weights_as_dict(model, weights))
    model.eval()

    n = A.shape[1]
    with torch.no_grad():
        agg_T, P_T, edge_weights = model(A, alpha)

    return agg_T, P_T, edge_weights


def loss_fcn(A, P_T):
    P = ns.lib.sparse_tensor.to_scipy(P_T)
    b = np.zeros(A.shape[1])
    r = np.random.RandomState(seed=0)
    x = r.randn(A.shape[1])
    x /= la.norm(x, 2)

    return ns.lib.multigrid.amg_2_v(A, P, b, x, res_tol=1e-10, singular=neumann_solve, jacobi_weight=omega)[1]


def fitness(generation, weights, weights_idx):
    r = np.random.RandomState(seed=generation)
    agg, P, edge_weights = compute_agg_and_p(weights, r)
    loss = loss_fcn(A, P)

    if np.isnan(loss.item()) or loss.item() == 1.:
        return 0
    else:
        return 1.0 / loss


def plot_grid(agg, P, bf_weights):
    graph = grid.networkx
    graph.remove_edges_from(list(nx.selfloop_edges(graph)))
    positions = {}
    for node in graph.nodes:
        positions[node] = grid.x[node]

    edge_values = np.zeros(len(graph.edges))
    for i, edge in enumerate(graph.edges):
        edge_values[i] = bf_weights[edge]

    if not isinstance(P, sp.spmatrix):
        P = ns.lib.sparse_tensor.to_scipy(P)

    plt.gcf().set_size_inches(figure_size[0], figure_size[1])
    nx.drawing.nx_pylab.draw_networkx(graph, ax=plt.gca(), pos=positions, arrows=False, with_labels=False, node_size=60, edge_color=edge_values, width=3.)
    grid.plot_agg(agg, alpha=0.1, edgecolor='0.2')
    # grid.plot_spider_agg(agg, P)
    plt.gca().set_aspect('equal')


def display_progress(ga_instance):
    weights = ga_instance.best_solution()
    gen = ga_instance.num_generation

    r = np.random.RandomState(seed=gen)
    agg, P, edge_weights = compute_agg_and_p(weights[0], r)
    agg = ns.lib.sparse_tensor.to_scipy(agg)
    loss = loss_fcn(A, P)

    print(f'Generation = {gen}')
    print(f'Fitness    = {weights[1]}')
    print(f'Loss       = {loss}')

    plt.clf()
    plot_grid(agg, P, edge_weights)
    plt.pause(0.1)

    model.load_state_dict(pygad.torchga.model_weights_as_dict(model, weights[0]))
    model.eval()
    # torch.save(model.state_dict(), f'models_chkpt/model_{gen:03}')


if __name__ == '__main__':
    baseline_conv = loss_fcn(A, ns.lib.sparse.to_torch_sparse(P_SA))
    print(f'Baseline convergence {baseline_conv:.4f}')

    if args.start_model is not None:
        model.load_state_dict(torch.load(args.start_model))
        model.eval()

    population = ns.ga.torch.TorchGA(model=model, num_solutions=args.initial_population_size)
    initial_population = population.population_weights

    plt.ion()
    plt.figure(figsize=figure_size)
    plot_grid(Agg, P_SA, A)
    plt.show()
    plt.pause(1)

    if greedy:
        perturb_val = 0.1
        selection='greedy'
        mutation_prob = 1.0
    else:
        perturb_val = 1.0
        selection='steady_state'
        mutation_prob=0.5

    perturb_val = 1
    num_workers = 1
    ga_instance = ns.ga.parga.ParallelGA(initial_population=initial_population,
                                         fitness_func=fitness,
                                         crossover_probability=0.5,
                                         selection=selection,
                                         mutation_probability=mutation_prob,
                                         mutation_min_perturb=-perturb_val,
                                         mutation_max_perturb=perturb_val,
                                         steady_state_top_use=2./3.,
                                         steady_state_bottom_discard=1./4.,
                                         num_workers=num_workers,
                                         model_folds=population.folds)
    ga_instance.start_workers()
    display_progress(ga_instance)

    for i in range(args.max_generations):
        if greedy:
            ga_instance.stochastic_iteration()
        else:
            ga_instance.iteration()
        display_progress(ga_instance)

    ga_instance.finish_workers()
