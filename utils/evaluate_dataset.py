import argparse

def parse_bool_str(v):
    v = v.lower()
    if v == 't' or v == 'true':
        return True
    else:
        return False

parser = argparse.ArgumentParser(description='Demo of the aggregate-picking network')
parser.add_argument('system', type=str, help='Problem in data folder to evaluate')
parser.add_argument('--model', type=str, help='Model file to evaluate')
parser.add_argument('--n', type=int, default=None, help='Size of the system.  In 2d, this determines the legnth in one dimension')
parser.add_argument('--alpha', type=float, default=0.1, help='Coarsening ratio for aggregation')
parser.add_argument('--workers', type=int, default=3, help='Number of workers to use for parallel evaluation of model')
parser.add_argument('--prefix', type=str, default='')
args = parser.parse_args()

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
import time
import pickle

sys.path.append('../')
import ns.model.agg_interp
import ns.model.data
import ns.lib.sparse
import ns.lib.sparse_tensor
import ns.lib.multigrid
import ns.parallel.pool
import common

N = args.n
alpha = args.alpha
omega = 2. / 3.
neumann_solve = False

def gridname_to_int(grid):
    fname = grid.extra['filename']
    fname = fname.split('/')[-1]
    fname = fname.split('.grid')[0]
    return int(fname)

ds = ns.model.data.Grid.load_dir(args.system)
ds.sort(key=gridname_to_int)
ds_size = list(map(lambda grid: grid.A.shape[0], ds))
extras = list(map(lambda grid: grid.extra, ds))

def evaluate_dataset(dataset, method='ml'):
    if not isinstance(method, str):
        raise ValueError('method is not a string')

    method_choices = ['ml', 'lloyd', 'dumb']
    if not method.lower() in method_choices:
        raise ValueError(f'method is not one of "{"".join(method_choices)}", got {method.lower()}')

    conv = np.zeros(len(dataset))
    for i in range(len(dataset)):
        A = dataset[i].A
        np.random.seed(0)
        if method == 'ml':
            with torch.no_grad():
                agg_T, P_T, bf_weights, cluster_centers, node_scores = model.forward(A, alpha)
            P = ns.lib.sparse_tensor.to_scipy(P_T)
        elif method == 'lloyd':
            C = common.strength_measure_funcs['olson'](A)
            Agg, _ = pyamg.aggregation.lloyd_aggregation(C, ratio=alpha, distance='same')
            P = ns.lib.multigrid.smoothed_aggregation_jacobi(A, Agg)
        else:
            rand = np.random.RandomState(0)
            N = A.shape[0]
            num_seeds = int(np.ceil(alpha * N))
            seeds = rand.permutation(N)[:num_seeds]
            C = common.strength_measure_funcs['olson'](A)
            seeds_T = torch.Tensor(seeds).long()

            distance, nearest_center = ns.lib.graph.modified_bellman_ford(ns.lib.sparse.scipy_to_torch(C), seeds_T)
            Agg_T = ns.lib.graph.nearest_center_to_agg(seeds_T, nearest_center)
            Agg = ns.lib.sparse.torch_to_scipy(Agg_T)
            P = ns.lib.multigrid.smoothed_aggregation_jacobi(A, Agg)
        b = np.zeros(A.shape[1])

        x = np.random.RandomState(0).randn(A.shape[1])
        x /= la.norm(x, 2)

        res = ns.lib.multigrid.amg_2_v(A, P, b, x, res_tol=1e-10, singular=neumann_solve, jacobi_weight=omega)[1]
        if np.isnan(res):
            conv[i] = 0.0
        else:
            conv[i] = res
    return conv


model = ns.model.agg_interp.FullAggNet(64, num_conv=2, iterations=4)
model.load_state_dict(torch.load(args.model))
model.eval()


def compute_test_loss_batch(index, method):
    batch = [ds[index]]
    return evaluate_dataset(batch, method).item()


def compute_test_loss(pool, dataset, method):
    convs = pool.map(np.arange(len(dataset)), compute_test_loss_batch, extra_args=(method,))
    return np.array(convs)


with ns.parallel.pool.WorkerPool(args.workers) as pool:
    print(f'Smallest problem: {np.min(ds_size)};  largest problem: {np.max(ds_size)}')

    print('Computing dumb baseline...')
    start = time.time()
    baseline = compute_test_loss(pool, ds, method='dumb')
    print(f'done in {time.time() - start:.3f} seconds')
    baseline_avg = np.average(baseline)
    print('Baseline avg', baseline_avg)

    print('Computing Lloyd, SA...')
    start = time.time()
    lloyd = compute_test_loss(pool, ds, method='lloyd')
    print(f'done in {time.time() - start:.3f} seconds')
    lloyd_avg = np.average(lloyd)
    print('Lloyd avg', lloyd_avg)

    print('Computing ML...')
    start = time.time()
    ml = compute_test_loss(pool, ds, method='ml')
    print(f'done in {time.time() - start:.3f} seconds')
    ml_avg = np.average(ml)
    print('ML avg', ml_avg)

    print(np.where(np.logical_and(ml > 0.93, lloyd > 0.93))[0])

    prefix = args.prefix
    if '3d' in args.system.lower():
        prefix += '3d_'

    if 'aniso' in args.system:
        prefix += 'aniso_'
    elif 'iso' in args.system:
        prefix += 'iso_'
    elif 'jump' in args.system:
        prefix += 'jump_'
    elif 'noisy' in args.system:
        prefix += 'noisy_'
    elif 'smooth' in args.system:
        prefix += 'smooth_'
    elif 'large' in args.system:
        prefix += 'large_'

    if 'train' in args.system:
        prefix += 'train'
    else:
        prefix += 'test'


    with open(f'data_out/{prefix}.pkl', 'wb') as f:
        pickle.dump({
            'baseline': baseline,
            'lloyd': lloyd,
            'ml': ml,
            'sizes': ds_size,
            'extras': extras
        }, f)
