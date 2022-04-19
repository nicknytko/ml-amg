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
import time
import pickle

sys.path.append('../')
import ns.model.agg_interp
import ns.model.data
import ns.lib.sparse
import ns.lib.sparse_tensor
import ns.lib.multigrid

import common

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
args = parser.parse_args()

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

print(f'Smallest problem: {np.min(ds_size)};  largest problem: {np.max(ds_size)}')

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

# def grab_extra(name):
#     def grabber(grid):
#         if name in grid.extra:
#             return grid.extra[name]
#         else:
#             return 0
#     return grabber

# thetas = list(map(grab_extra('theta'), ds))
# epsilons = list(map(grab_extra('epsilon'), ds))

print('Computing dumb baseline...')
start = time.time()
baseline = evaluate_dataset(ds, method='dumb')
print(f'done in {time.time() - start:.3f} seconds')
baseline_avg = np.average(baseline)
print('Baseline avg', baseline_avg)


print('Computing Lloyd, SA...')
start = time.time()
lloyd = evaluate_dataset(ds, method='lloyd')
print(f'done in {time.time() - start:.3f} seconds')
lloyd_avg = np.average(lloyd)
print('Lloyd avg', lloyd_avg)


print('Computing ML...')
start = time.time()
ml = evaluate_dataset(ds, method='ml')
print(f'done in {time.time() - start:.3f} seconds')
ml_avg = np.average(ml)
print('ML avg', ml_avg)


print(np.where(np.logical_and(ml > 0.93, lloyd > 0.93))[0])


prefix = ''
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


#iso_prefix = ('Anisotropic' if 'aniso' in args.system else 'Isotropic')
with open(f'data_out/{prefix}.pkl', 'wb') as f:
    pickle.dump({
        'baseline': baseline,
        'lloyd': lloyd,
        'ml': ml,
        'sizes': ds_size,
        'extras': extras
    }, f)


# plt.figure(figsize=(8,8))
# plt.title(f'{iso_prefix}: ML vs Lloyd Convergence Analysis')
# plt.scatter(lloyd, ml, s=np.array(ds_size)**0.8, alpha=0.7, label='Convergence')
# plt.plot([0, 1], [0, 1], 'tab:orange', label='Diagonal')
# plt.plot([0, 1], [ml_avg, ml_avg], 'tab:green', label='ML Average')
# plt.plot([lloyd_avg, lloyd_avg], [0, 1], 'tab:red', label='Lloyd Average')
# plt.xlim((0, 1))
# plt.ylim((0, 1))
# plt.xlabel('Lloyd Convergence')
# plt.ylabel('ML Convergence')
# plt.axis('equal')
# plt.grid()
# plt.legend()
# plt.savefig(f'{prefix}_lloyd_ml_convergence.pdf')

# plt.figure(figsize=(8,8))
# plt.title(f'{iso_prefix}: Convergence of methods vs Problem Size')
# plt.semilogx(ds_size, lloyd, 'o', label='Lloyd Convergence')
# plt.semilogx(ds_size, ml, 'o', label='ML Convergence')
# plt.semilogx(ds_size, baseline, 'o', label='Baseline Convergence')
# plt.xlabel('Problem Size (DOF)')
# plt.ylabel('Convergence Rate')
# plt.grid()
# plt.legend()
# plt.savefig(f'{prefix}_convergence_per_size.pdf')

# plt.figure(figsize=(8,8))
# plt.title(f'{iso_prefix}: Relative performance of ML to Lloyd')
# plt.semilogx(ds_size, ml / lloyd, 'o', label='Relative Performance')
# plt.xlabel('Problem Size (DOF)')
# plt.ylabel('Ratio of ML to Lloyd convergence')
# plt.grid()
# plt.legend()
# plt.savefig(f'rel_perf_{prefix}.pdf')

# if 'theta' in ds[0].extra and 'epsilon' in ds[0].extra:
#     plt.figure(figsize=(8,8))
#     plt.title(f'{iso_prefix}: Convergence of Lloyd and ML vs Magnitude of Anisotropy')
#     plt.semilogx(epsilons, baseline, 'o', label='Baseline Convergence')
#     plt.semilogx(epsilons, ml, 'o', label='ML Convergence')
#     plt.xlabel('Epsilon')
#     plt.ylabel('Convergence Rate')
#     plt.grid()
#     plt.legend()
#     plt.savefig(f'{prefix}_convergence_per_epsilon.pdf')

#     plt.figure(figsize=(8,8))
#     plt.title(f'{iso_prefix}: Convergence of Lloyd and ML vs Rotation of Anisotropy')
#     plt.plot(thetas, baseline, 'o', label='Baseline Convergence')
#     plt.plot(thetas, ml, 'o', label='ML Convergence')
#     plt.xlabel('Theta')
#     plt.ylabel('Convergence Rate')
#     plt.grid()
#     plt.legend()
#     plt.savefig(f'{prefix}_convergence_per_theta.pdf')

# plt.show()
