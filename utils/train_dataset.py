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

import common

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

train = ns.model.data.Grid.load_dir(os.path.join(args.system, 'train'))[::8]
test = ns.model.data.Grid.load_dir(os.path.join(args.system, 'test'))[::8]
print(len(train), len(test))

model = ns.model.agg_interp.AggOnlyNet(80)
model_folds = [
    'AggNet.layers.0',
    'AggNet.layers.1',
    'AggNet.layers.2',
    'AggNet.layers.3',
    'AggNet.edge_out',
    # 'PNet'
]

def fitness(generation, weights, weights_idx):
    return 1 - common.evaluate_dataset(weights, train, model, alpha=args.alpha, gen=generation)

def display_progress(ga_instance):
    weights, fitness, _ = ga_instance.best_solution()
    gen = ga_instance.num_generation
    test_loss = common.evaluate_dataset(weights, test, model, alpha=args.alpha, gen=gen)

    print(f'Generation = {gen}')
    print(f'Train Loss = {1.0 - fitness}')
    print(f'Test Loss = {test_loss}')

    writer.add_scalars('Loss/Train', {'ML': 1 - fitness, 'Lloyd/SA': train_benchmark}, gen)
    writer.add_scalars('Loss/Test', {'ML': test_loss, 'Lloyd/SA': test_benchmark}, gen)

    model.load_state_dict(ns.ga.torch.model_weights_as_dict(model, weights))
    model.eval()
    torch.save(model.state_dict(), f'models_chkpt/model_{gen:03}')

if __name__ == '__main__':
    writer = tensorboard.SummaryWriter()

    S = common.strength_measure_funcs[args.strength_measure]
    train_benchmark = np.average(common.evaluate_ref_conv(train, S, alpha=args.alpha)); print(f'Evaluated train benchmark ({train_benchmark:.4f})')
    test_benchmark = np.average(common.evaluate_ref_conv(test, S, alpha=args.alpha)); print(f'Evaluated test benchmark ({test_benchmark:.4f})')

    try:
        os.mkdir('models_chkpt')
    except:
        pass

    if args.start_model is not None:
        model.load_state_dict(torch.load(args.start_model))
        model.eval()
    population = ns.ga.torch.TorchGA(model=model, num_solutions=args.initial_population_size, model_fold_names=model_folds)
    initial_population = population.population_weights

    perturb_val = 5.
    ga_instance = ns.ga.parga.ParallelGA(initial_population=initial_population,
                                         fitness_func=fitness,
                                         crossover_probability=0.5,
                                         mutation_probability=0.4,
                                         mutation_min_perturb=-perturb_val,
                                         mutation_max_perturb=perturb_val,
                                         steady_state_top_use=2./3.,
                                         steady_state_bottom_discard=1./4.,
                                         num_workers=args.workers,
                                         model_folds=population.folds)
    ga_instance.num_generation = args.start_generation
    ga_instance.start_workers()
    display_progress(ga_instance)

    for i in range(args.max_generations):
        ga_instance.iteration()
        display_progress(ga_instance)

    ga_instance.finish_workers()
