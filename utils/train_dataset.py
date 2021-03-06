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
import traceback
import time

sys.path.append('../')
import ns.model.agg_interp
import ns.model.data
import ns.lib.sparse
import ns.lib.sparse_tensor
import ns.lib.multigrid
import ns.lib.graph
import ns.ga.parga
import ns.ga.torch
import ns.parallel.pool
import ns.lib.profiler

import common

parser = argparse.ArgumentParser(description='Demo of the aggregate-picking network')
parser.add_argument('system', type=str, help='Problem in data folder to train on')
parser.add_argument('--max-generations', type=int, default=500, help='Maximum number of training generations')
parser.add_argument('--population-size', type=int, default=20, help='Initial population size')
parser.add_argument('--alpha', type=float, default=0.1, help='Coarsening ratio for aggregation')
parser.add_argument('--workers', type=int, default=3, help='Number of workers to use for parallel GA training')
parser.add_argument('--start-generation', type=int, default=0, help='Initial generation (used for resuming training)')
parser.add_argument('--start-model', type=str, default=None, help='Initial generation (used for resuming training)')
parser.add_argument('--strength-measure', default='abs', choices=common.strength_measure_funcs.keys())
parser.add_argument('--greedy', default=False, type=common.parse_bool_str)
parser.add_argument('--batched', default=False, type=common.parse_bool_str)
parser.add_argument('--compute-test-loss', default=True, type=common.parse_bool_str)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--loss-relative-measure', type=common.parse_bool_str, default=True)
parser.add_argument('--cuda', type=common.parse_bool_str, default=False)
parser.add_argument('--evaluate-bench-loss', type=common.parse_bool_str, default=True)
parser.add_argument('--pre-smooth', type=int, default=1)
parser.add_argument('--post-smooth', type=int, default=1)

ns.lib.profiler.Profiler.enabled = True

args = parser.parse_args()

greedy = args.greedy
batched = args.batched
num_pre_relax = args.pre_smooth
num_post_relax = args.post_smooth

# Use train/ and test/ dir if they exist in the system folder.  Otherwise, train=test=system
train_dir = os.path.join(args.system, 'train')
test_dir = os.path.join(args.system, 'test')
if not os.path.exists(train_dir) or not os.path.exists(test_dir):
    train_dir = args.system
    test_dir = args.system

train = ns.model.data.Grid.load_dir(train_dir)
test = ns.model.data.Grid.load_dir(test_dir)

# Benchmark loss -- initialized to dummy values for now
S = common.strength_measure_funcs[args.strength_measure]
train_benchmark = np.ones(len(train))
test_benchmark = train_benchmark

device = ('cuda' if args.cuda else 'cpu')
model = ns.model.agg_interp.FullAggNet(8, num_conv=2, iterations=2).to(device)
batch_size = args.batch_size
writer = None

def evaluate_dataset(weights, dataset, model=None, alpha=0.3, omega=2./3.):
    model.load_state_dict(ns.ga.torch.model_weights_as_dict(model, weights))
    model.eval()

    conv = torch.zeros(len(dataset))
    for i in range(len(dataset)):
        A = dataset[i].A
        if args.cuda:
            A_T = ns.lib.sparse.scipy_to_torch(A).to(model.device)
        n = A.shape[1]
        b = np.zeros(n)
        P = None

        try:
            with torch.no_grad():
                P_T = None
                with ns.lib.profiler.Profiler('model inferencing'):
                    agg_T, P_T, bf_weights, cluster_centers, node_scores = model.forward(A, alpha)
                P = ns.lib.sparse.torch_to_scipy(P_T)
        except Exception as e:
            print(f'Could not evaluate grid {i}: {traceback.format_exc()}')
            conv[i] = 1.0
            continue

        x = np.random.RandomState(0).randn(A.shape[1])
        x /= la.norm(x, 2)
        #if args.cuda:
        if False:
            b = torch.zeros(A.shape[1]).to(model.device)
            x = torch.Tensor(x).to(model.device)
            res = ns.lib.multigrid.amg_2_v_torch(A_T, P_T, b, x, error_tol=1e-6, jacobi_weight=omega, pre_smoothing_steps=num_pre_relax, post_smoothing_steps=num_post_relax)
        else:
            b = np.zeros(A.shape[1])
            res = ns.lib.multigrid.amg_2_v(A, P, b, x, error_tol=1e-6, pre_smoothing_steps=num_pre_relax, post_smoothing_steps=num_post_relax)[1]
        conv[i] = res
    conv[torch.isnan(conv)] = 1.
    return conv


def fitness(generation, weights, weights_idx):
    if batched:
        if batch_size < len(train):
            rand = np.random.RandomState(generation)
            batch_indices = rand.choice(len(train), size=batch_size, replace=False)
            batch = [train[i] for i in batch_indices]
            batch_ref_conv = train_benchmark[batch_indices]
        else:
            batch = train
            batch_ref_conv = train_benchmark
    else:
        batch = train
        batch_ref_conv = train_benchmark

    raw_conv = evaluate_dataset(weights, batch, model, alpha=args.alpha)
    if args.loss_relative_measure:
        return 1./np.average(raw_conv / batch_ref_conv)
    else:
        return 1./np.average(raw_conv)


def compute_test_loss_batch(index, generation, weights):
    batch = [test[index]]
    raw_conv = common.evaluate_dataset(weights, batch, model, alpha=args.alpha, gen=generation).item()

    if args.loss_relative_measure:
        return raw_conv / test_benchmark[index]
    else:
        return raw_conv


def compute_test_loss(ga_instance, generation, weights):
    convs = ga_instance.parallel_map(np.arange(len(test)), compute_test_loss_batch, extra_args=(generation, weights))
    convs = np.mean(np.array(convs))
    return convs


def compute_ref_loss_batch(index, ds):
    global S
    ds = (train if ds==1 else test)
    batch = [ds[index]]
    return common.evaluate_ref_conv(batch, S, alpha=args.alpha).item()


def compute_ref_loss(pool, ds='train'):
    ds = (1 if ds=='train' else 0)
    ds_len = (len(train) if ds==1 else len(test))
    return np.array(pool.map(np.arange(ds_len), compute_ref_loss_batch, extra_args=(ds, )))


def display_progress(ga_instance):
    global writer
    weights, fitness, _ = ga_instance.best_solution()
    gen = ga_instance.num_generation

    # Get training batch used
    if batched:
        rand = np.random.RandomState(gen)
        batch_indices = rand.choice(len(train), size=batch_size, replace=False)
        batch = [train[i] for i in batch_indices]
        batch_ref_conv = train_benchmark[batch_indices]
    else:
        batch = train
        batch_ref_conv = train_benchmark

    # Compute test loss
    if args.compute_test_loss:
        test_loss = compute_test_loss(ga_instance, gen, weights)
    else:
        test_loss = 1. / fitness

    # Get benchmark loss
    if args.loss_relative_measure:
        lloyd_train = 1.
        lloyd_test = 1.
    else:
        lloyd_train = np.average(batch_ref_conv)
        lloyd_test = np.average(test_benchmark)

    # Finally, output everything
    print(f'Generation = {gen}')
    print(f'Train Loss = {1.0 / fitness}')
    print(f'Test Loss = {test_loss}')

    writer.add_scalars('Loss/Train', {'ML': 1/fitness, 'Lloyd': lloyd_train}, gen)
    writer.add_scalars('Loss/Test', {'ML': test_loss, 'Lloyd': lloyd_test}, gen)
    writer.add_scalars('Population Training Loss', dict(zip(map(lambda x: str(x), range(ga_instance.population_size)),
                                                            1./np.sort(ga_instance.population_fitness))), gen)

    model.load_state_dict(ns.ga.torch.model_weights_as_dict(model, weights))
    model.eval()
    torch.save(model.state_dict(), f'models_chkpt/model_{gen:03}')

    cur_pop_fit = np.sort(1-ga_instance.population_fitness)


with ns.parallel.pool.WorkerPool(args.workers) as pool:
    writer = tensorboard.SummaryWriter('runs')

    # Compute benchmark Lloyd loss
    if args.evaluate_bench_loss:
        train_benchmark = compute_ref_loss(pool, 'train')
        if args.compute_test_loss:
            test_benchmark = compute_ref_loss(pool, 'test')
        else:
            test_benchmark = train_benchmark
    print(f'Evaluated train benchmark ({np.average(train_benchmark):.4f})')
    print(f'Evaluated test benchmark ({np.average(test_benchmark):.4f})')

    # Create model checkpoint folder
    try:
        os.mkdir('models_chkpt')
    except:
        pass

    # Load starting model if we have one specified
    if args.start_model is not None:
        model.load_state_dict(torch.load(args.start_model))
        model.eval()

    # Seed our population
    population = ns.ga.torch.TorchGA(model=model, num_solutions=args.population_size)
    initial_population = population.population_weights

    # Specify parameters for GA
    if greedy:
        perturb_val = 0.5
        selection='greedy'
        mutation_prob = 1.0
    else:
        perturb_val = 0.5
        selection='steady_state'
        mutation_prob=1.0

    ga_instance = ns.ga.parga.ParallelGA(initial_population=initial_population,
                                         fitness_func=fitness,
                                         crossover_probability=0.0,
                                         selection=selection,
                                         mutation_probability=mutation_prob,
                                         mutation_min_perturb=-perturb_val,
                                         mutation_max_perturb=perturb_val,
                                         steady_state_top_use=1./2.,
                                         steady_state_bottom_discard=1./2.,
                                         worker_pool=pool,
                                         model_folds=population.folds)
    ga_instance.num_generation = args.start_generation
    display_progress(ga_instance)

    for i in range(args.max_generations):
        if batched and batch_size < len(train):
            ga_instance.stochastic_iteration()
        else:
            ga_instance.iteration()
        display_progress(ga_instance)
