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
args = parser.parse_args()

neumann_solve = False
alpha = 0.3
omega = 2. / 3.

train = ns.model.data.Grid.load_dir(os.path.join(args.system, 'train'))[::8]
test = ns.model.data.Grid.load_dir(os.path.join(args.system, 'test'))[::4]

writer = tensorboard.SummaryWriter()

def evaluate_dataset(weights, dataset, use_model=True):
    if use_model:
        model = ns.model.agg_interp.FullAggNet(64, use_aggnet=True, use_pnet=True)
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


train_benchmark = evaluate_dataset(None, train, False)
test_benchmark = evaluate_dataset(None, test, False)


def fitness(weights, weights_idx):
    conv = evaluate_dataset(weights, train)
    return 1 - conv


def display_progress(ga_instance):
    weights, fitness, _ = ga_instance.best_solution()
    gen = ga_instance.generations_completed
    test_loss = evaluate_dataset(weights, test)

    print(f'Generation = {gen}')
    print(f'Train Loss = {1.0 - fitness}')
    print(f'Test Loss = {test_loss}')

    writer.add_scalars('Loss/Train', {'ML': 1 - fitness, 'Lloyd/SA': train_benchmark}, gen)
    writer.add_scalars('Loss/Test', {'ML': test_loss, 'Lloyd/SA': test_benchmark}, gen)

    model = ns.model.agg_interp.FullAggNet(64, use_aggnet=True, use_pnet=True)
    model.load_state_dict(pygad.torchga.model_weights_as_dict(model, weights))
    model.eval()
    torch.save(model.state_dict(), f'models_chkpt/model_{gen:03}')


try:
    os.mkdir('models_chkpt')
except:
    pass

if __name__ == '__main__':
    model = ns.model.agg_interp.FullAggNet(64, use_aggnet=True, use_pnet=True)
    initial_population = pygad.torchga.TorchGA(model=model, num_solutions=args.initial_population_size).population_weights
    print(initial_population)
    ga_instance = pygad.GA(num_generations=args.max_generations,
                           num_parents_mating=15,
                           initial_population=initial_population,
                           fitness_func=fitness,
                           on_generation=display_progress,
                           mutation_probability=0.4,
                           random_mutation_min_val=-3.,
                           random_mutation_max_val=3.,
                           keep_parents=1)
    display_progress(ga_instance)
    ga_instance.run()
