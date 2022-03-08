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
parser.add_argument('--greedy', default=False, type=common.parse_bool_str)
args = parser.parse_args()

greedy = args.greedy
train = ns.model.data.Grid.load_dir(os.path.join(args.system, 'train'))[:]
test = ns.model.data.Grid.load_dir(os.path.join(args.system, 'test'))[:]
print(len(train), len(test))

model = ns.model.agg_interp.FullAggNet(64, num_conv=2, iterations=4)

batch_size = 64
def fitness(generation, weights, weights_idx):
    if greedy:
        rand = np.random.RandomState(generation)
        batch_indices = rand.choice(len(train), size=batch_size, replace=False)
        batch = [train[i] for i in batch_indices]
    else:
        batch = train[::8]
    return 1 - common.evaluate_dataset(weights, batch, model, alpha=args.alpha, gen=generation)


def compute_test_loss_batch(index, generation, weights):
    batch = [test[index]]
    return common.evaluate_dataset(weights, batch, model, alpha=args.alpha, gen=generation)


def compute_test_loss(ga_instance, generation, weights):
    convs = ga_instance.parallel_map(np.arange(len(test)), compute_test_loss_batch, extra_args=(generation, weights))
    convs = np.mean(np.array(convs))
    return convs


def display_progress(ga_instance):
    weights, fitness, _ = ga_instance.best_solution()
    gen = ga_instance.num_generation
    test_loss = compute_test_loss(ga_instance, gen, weights)

    print(f'Generation = {gen}')
    print(f'Train Loss = {1.0 - fitness}')
    print(f'Test Loss = {test_loss}')

    writer.add_scalars('Loss/Train', {'ML': 1 - fitness, 'Lloyd/SA': train_benchmark}, gen)
    writer.add_scalars('Loss/Test', {'ML': test_loss, 'Lloyd/SA': test_benchmark}, gen)

    model.load_state_dict(ns.ga.torch.model_weights_as_dict(model, weights))
    model.eval()
    torch.save(model.state_dict(), f'models_chkpt/model_{gen:03}')

    cur_pop_fit = np.sort(1-ga_instance.population_fitness)

if __name__ == '__main__':
    writer = tensorboard.SummaryWriter('runs')

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

    population = ns.ga.torch.TorchGA(model=model, num_solutions=args.initial_population_size)#, model_fold_names=model_folds)
    initial_population = population.population_weights

    if greedy:
        perturb_val = 0.1
        selection='greedy'
        mutation_prob = 1.0
    else:
        perturb_val = 1.0
        selection='steady_state'
        mutation_prob=0.5

    ga_instance = ns.ga.parga.ParallelGA(initial_population=initial_population,
                                         fitness_func=fitness,
                                         crossover_probability=0.5,
                                         selection=selection,
                                         mutation_probability=mutation_prob,
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
        if greedy:
            ga_instance.stochastic_iteration()
        else:
            ga_instance.iteration()
        display_progress(ga_instance)

    ga_instance.finish_workers()
