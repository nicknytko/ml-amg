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
import ns.lib.aggplot
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
parser.add_argument('--start-model', type=str, default=None, help='Initial generation (used to seed grid)')
args = parser.parse_args()

neumann_solve = False
alpha = args.alpha

greedy = args.greedy

if os.path.exists(args.system):
    if alpha is None:
        alpha = 1. / 3.
    grid = ns.model.data.Grid.load(args.system)
    A = grid.A
    n = A.shape[0]
    k = int(np.ceil(alpha*n))
    np.random.seed(0)
    C = common.strength_measure_funcs['olson'](A)
    #Agg, Agg_roots = pyamg.aggregation.lloyd_aggregation(C, ratio=alpha, distance='same')
    Agg, Agg_roots, seeds = ns.lib.graph.lloyd_aggregation(C, ratio=alpha, distance='same')
    figure_size = (10,10)
else:
    raise RuntimeError(f'No system {args.system}')
N = n

np.random.seed()

A_T = ns.lib.sparse.to_torch_sparse(A)

# Set up Jacobi smoother
Dinv = sp.diags([1.0 / A.diagonal()], [0])
omega = (4. / 3.) / np.abs(spla.eigs(Dinv @ A, k=1, return_eigenvectors=False)).item()
smoother = (sp.eye(n) - omega*Dinv@A)
P_SA = smoother @ Agg

# Create PyTorch tensor versions of everything
smoother_T = ns.lib.sparse.to_torch_sparse(smoother)
Agg_T = ns.lib.sparse.to_torch_sparse(Agg)
Agg_roots_T = torch.Tensor(Agg_roots)
A_Graph = ns.model.data.graph_from_matrix_basic(A)
agg_v = np.zeros(A.shape[1])
agg_v[Agg_roots] = 1.

plot = None
err_plot = None


class ProblemParamContainer(nn.Module):
    def __init__(self, grid, alpha):
        super(ProblemParamContainer, self).__init__()
        self.grid = grid
        n = grid.A.shape[0]
        k = int(np.ceil(n * alpha))
        self.n = n
        self.k = k
        self.soft_assignment = nn.Parameter(torch.Tensor(np.array(Agg.todense())), requires_grad=False)
        for i in range(n):
            self.soft_assignment[i] = ns.model.agg_interp.topk_vec(self.soft_assignment[i], 1)
        # self.mu = nn.Parameter(torch.rand((n, k)), requires_grad=False)
        # self.lmbda = nn.Parameter(torch.rand(n), requires_grad=False)

    def hard_assignment(self):
        soft_numpy = self.soft_assignment.numpy()
        cols = np.argmax(soft_numpy, axis=1)
        Agg_coo = sp.coo_matrix((np.ones(self.n), (np.arange(self.n), cols)), shape=(self.n, self.k))
        return Agg_coo.tocsr()

    def forward(self):
        Agg = self.hard_assignment()
        P = smoother@Agg
        return self.soft_assignment.numpy(), Agg, P

model = ProblemParamContainer(grid, alpha)


def loss_fcn(A, P):
    b = np.zeros(A.shape[1])
    r = np.random.RandomState(seed=0)
    x = r.randn(A.shape[1])
    x /= la.norm(x, 2)

    mg_loss = ns.lib.multigrid.amg_2_v(A, P, b, x, res_tol=1e-10, singular=neumann_solve, jacobi_weight=omega)[1]
    return mg_loss


def fitness(generation, weights, weights_idx):
    model.load_state_dict(ns.ga.torch.model_weights_as_dict(model, weights))
    model.eval()

    SoftAgg, Agg, P = model()

    # if not ns.lib.graph.check_aggregates_connected(A, Agg):
    #     return 0

    loss = loss_fcn(A, P)#, SoftAgg, model.mu.numpy(), model.lmbda.numpy())
    if np.isnan(loss.item()) or loss.item() == 1.:
        return 0
    else:
        return 1./loss


def display_progress(weights, gen, loss):
    model.load_state_dict(ns.ga.torch.model_weights_as_dict(model, weights))
    model.eval()
    SoftAgg, agg, P = model()

    print(f'Generation = {gen}')
    print(f'Fitness    = {1./loss}')
    print(f'Loss       = {loss}')

    plot.clear()
    plot.plot_agg_3d_voxel(grid, agg)
    plot.set_title(f'Generation {gen}, GA Optimization AMG, conv={loss:.4f}')


def display_progress_ga(ga_instance):
    weights = ga_instance.best_solution()
    gen = ga_instance.num_generation
    loss = 1./weights[1]
    display_progress(weights[0], gen, loss)


def custom_crossover(self):
    N_to_replace = int(self.steady_state_bottom_discard * self.population_size)
    N_top_to_use = int(self.steady_state_top_use * self.population_size)
    chromosomes = self.population.shape[1]
    population_to_use = np.argsort(self.population_fitness)[:N_top_to_use]

    random = np.random

    created_population = []
    while len(created_population) < N_to_replace:
        parent_one_idx, parent_two_idx = random.choice(population_to_use, size=2, replace=False)
        parent_one = self.population[parent_one_idx]
        parent_two = self.population[parent_two_idx]
        p = random.rand()
        if p <= self.crossover_probability:
            child_one = np.zeros(chromosomes)
            child_two = np.zeros(chromosomes)

            for i in range(N):
                if random.choice([True, False]):
                    child_one[i*k:(i+1)*k] = parent_one[i*k:(i+1)*k]
                    child_two[i*k:(i+1)*k] = parent_two[i*k:(i+1)*k]
                else:
                    child_one[i*k:(i+1)*k] = parent_two[i*k:(i+1)*k]
                    child_two[i*k:(i+1)*k] = parent_one[i*k:(i+1)*k]

                created_population.append(child_one)
                created_population.append(child_two)
        else:
            created_population.append(parent_one)
            created_population.append(parent_two)

    # Pick worst fit individuals to replace
    indices_to_replace = np.argsort(self.population_fitness)[:N_to_replace]

    # We will compute more pairs than needed, then discard a random subset
    indices_to_use = np.random.choice(np.arange(0, len(created_population)), size=N_to_replace, replace=False)
    created_population = np.array(created_population)

    # Replace subset of population
    self.population[indices_to_replace] = created_population[indices_to_use]
    self.population_computed_fitness[indices_to_replace] = False


def mat_to_adj(A):
    return sp.csr_matrix((np.ones_like(A.data), A.indices, A.indptr), shape=A.shape)


def custom_mutation(self):
    N_pop = self.population.shape[0]
    chromosomes = self.population.shape[1]
    random = np.random

    Adj = mat_to_adj(A)

    for i in range(N_pop):
        # if self.population_computed_fitness[i]:
        #     continue # Only mutate new offspring

        temp_model = ProblemParamContainer(grid, alpha)
        temp_model.load_state_dict(ns.ga.torch.model_weights_as_dict(temp_model, self.population[i]))
        temp_model.eval()
        Agg = temp_model.hard_assignment()

        mp = random.rand()
        #mp = self.mutation_probability

        for j in range(N):
            if random.rand() <= mp:
                agg_conn = Adj@Agg.tocsc()
                potential_aggs = agg_conn[j].nonzero()[1]
                new_agg = random.choice(potential_aggs)
                self.population[i, j*k:(j+1)*k] = 0
                self.population[i, j*k+new_agg] = 1
                self.population_computed_fitness[i] = False



if __name__ == '__main__':
    baseline_conv = loss_fcn(A, P_SA)
    print(f'Baseline convergence {baseline_conv:.4f}')

    plot_ref = ns.lib.aggplot.ThreadedPlot()
    plot_ref.create_axes(projection='3d')
    plot_ref.plot_agg_3d_voxel(grid, Agg)
    plot_ref.set_title(f'Lloyd Aggregation, conv={baseline_conv:.4f}')

    plot = ns.lib.aggplot.ThreadedPlot()
    plot.create_axes(projection='3d')

    gs_err_plot = ns.lib.aggplot.ThreadedPlot()
    gs_err_plot.create_axes(projection='3d')
    uv = np.random.randn(n)
    L = sp.tril(grid.A).tocsr()
    U = sp.triu(grid.A, k=1).tocsr()
    nu = 10
    for i in range(nu):
        uv = spla.spsolve_triangular(L, -U@uv)
    gs_err_plot.plot_3d_grid_voxel(grid, np.abs(uv))
    gs_err_plot.set_title('Gauss-Seidel error after 10 iterations')
    gs_err_plot.set_xlabel('x')
    gs_err_plot.set_ylabel('y')
    gs_err_plot.set_zlabel('z')

    if args.start_model is not None:
        start_model = ns.model.agg_interp.FullAggNet(64, num_conv=2, iterations=4)
        start_model.load_state_dict(torch.load(args.start_model))
        start_model.eval()
        agg_T, P_T, C_T, top_k, node_scores = start_model(A, alpha)
        model.soft_assignment.copy_(agg_T.to_dense())

    population = ns.ga.torch.TorchGA(model=model, num_solutions=args.initial_population_size, random_perturb=0.)
    initial_population = population.population_weights

    if greedy:
        perturb_val = 0.05
        selection='greedy'
        mutation_prob = 1.0
    else:
        perturb_val = 0.05
        #selection=custom_crossover
        selection='greedy'
        mutation=custom_mutation
        mutation_prob=0.1

    num_workers = 3
    ga_instance = ns.ga.parga.ParallelGA(initial_population=initial_population,
                                         fitness_func=fitness,
                                         crossover_probability=0.5,
                                         selection=selection,
                                         mutation=mutation,
                                         mutation_probability=mutation_prob,
                                         mutation_min_perturb=-perturb_val,
                                         mutation_max_perturb=perturb_val,
                                         steady_state_top_use=2./3.,
                                         steady_state_bottom_discard=1./4.,
                                         num_workers=num_workers,
                                         model_folds=population.folds)
    ga_instance.start_workers()
    display_progress_ga(ga_instance)

    #for i in range(args.max_generations):
    while True:
        if greedy:
            ga_instance.stochastic_iteration()
        else:
            ga_instance.iteration()
        display_progress_ga(ga_instance)

    ga_instance.finish_workers()
