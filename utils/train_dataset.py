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
import ns.ga.parga

def parse_bool_str(v):
    v = v.lower()
    if v == 't' or v == 'true':
        return True
    else:
        return False

parser = argparse.ArgumentParser(description='Demo of the aggregate-picking network')
parser.add_argument('system', type=str, help='Problem in data folder to train on')
parser.add_argument('--max-generations', type=int, default=500, help='Maximum number of training generations')
parser.add_argument('--initial-population-size', type=int, default=20, help='Initial population size')
parser.add_argument('--alpha', type=float, default=None, help='Coarsening ratio for aggregation')
parser.add_argument('--workers', type=int, default=4, help='Number of workers to use for parallel GA training')
args = parser.parse_args()

neumann_solve = False
alpha = 0.3
omega = 2. / 3.

train = ns.model.data.Grid.load_dir(os.path.join(args.system, 'train'))[::8]
test = ns.model.data.Grid.load_dir(os.path.join(args.system, 'test'))[::4]
model = ns.model.agg_interp.FullAggNet(64)

def evaluate_dataset(weights, dataset, use_model=True):
    if use_model:
        model.load_state_dict(pygad.torchga.model_weights_as_dict(model, weights))
        model.eval()

    conv = np.zeros(len(dataset))
    for i in range(len(dataset)):
        A = dataset[i].A
        if use_model:
            with torch.no_grad():
                agg_T, P_T, bf_weights, cluster_centers, node_scores = model.forward(A, alpha)
            P = ns.lib.sparse_tensor.to_scipy(P_T)
        else:
            Agg, _ = pyamg.aggregation.lloyd_aggregation(A, ratio=alpha)
            P = ns.lib.multigrid.smoothed_aggregation_jacobi(A, Agg)
        b = np.zeros(A.shape[1])

        np.random.seed(0)
        x = np.random.randn(A.shape[1])
        x /= la.norm(x, 2)
        np.random.seed()

        res = ns.lib.multigrid.amg_2_v(A, P, b, x, res_tol=1e-10, singular=neumann_solve, jacobi_weight=omega)[1]
        if np.isnan(res):
            conv[i] = 0.0
        else:
            conv[i] = res
    return np.average(conv)


def fitness(weights, weights_idx):
    conv = evaluate_dataset(weights, train)
    return 1 - conv


def display_progress(ga_instance):
    weights, fitness, _ = ga_instance.best_solution()
    gen = ga_instance.num_generation
    test_loss = evaluate_dataset(weights, test)

    print(f'Generation = {gen}')
    print(f'Train Loss = {1.0 - fitness}')
    print(f'Test Loss = {test_loss}')

    writer.add_scalars('Loss/Train', {'ML': 1 - fitness, 'Lloyd/SA': train_benchmark}, gen)
    writer.add_scalars('Loss/Test', {'ML': test_loss, 'Lloyd/SA': test_benchmark}, gen)

    model.load_state_dict(pygad.torchga.model_weights_as_dict(model, weights))
    model.eval()
    torch.save(model.state_dict(), f'models_chkpt/model_{gen:03}')


if __name__ == '__main__':
    writer = tensorboard.SummaryWriter()
    train_benchmark = evaluate_dataset(None, train, False)
    test_benchmark = evaluate_dataset(None, test, False)

    try:
        os.mkdir('models_chkpt')
    except:
        pass

    initial_population = np.array(pygad.torchga.TorchGA(model=model, num_solutions=args.initial_population_size).population_weights)
    ga_instance = ns.ga.parga.ParallelGA(initial_population=initial_population,
                                         fitness_func=fitness,
                                         crossover_probability=0.5,
                                         mutation_probability=0.4,
                                         mutation_min_perturb=-3.,
                                         mutation_max_perturb=3.,
                                         steady_state_top_use=2./3.,
                                         steady_state_bottom_discard=1./4.,
                                         num_workers=args.workers)
    ga_instance.start_workers()
    display_progress(ga_instance)

    for i in range(args.max_generations):
        ga_instance.iteration()
        display_progress(ga_instance)

    ga_instance.finish_workers()
