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

sys.path.append('../')
import ns.model.agg_interp
import ns.model.data
import ns.lib.sparse
import ns.lib.sparse_tensor
import ns.lib.multigrid

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
parser.add_argument('--alpha', type=float, default=0.3, help='Coarsening ratio for aggregation')
args = parser.parse_args()

N = args.n
alpha = args.alpha
omega = 2. / 3.
neumann_solve = False

ds = ns.model.data.Grid.load_dir(args.system)
ds_size = list(map(lambda grid: grid.A.shape[0], ds))

print(f'Smallest problem: {np.min(ds_size)};  largest problem: {np.max(ds_size)}')

def evaluate_dataset(dataset, use_model=True):
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
    return conv

model = ns.model.agg_interp.FullAggNet(64)
model.load_state_dict(torch.load(args.model))
model.eval()

print('Computing Lloyd, SA baseline...')
start = time.time()
baseline = evaluate_dataset(ds, False)
print(f'done in {time.time() - start:.3f} seconds')

print('Computing ML...')
start = time.time()
ml = evaluate_dataset(ds, True)
print(f'done in {time.time() - start:.3f} seconds')

baseline_avg = np.average(baseline)
ml_avg = np.average(ml)

plt.figure(figsize=(8,8))
plt.title('ML vs Lloyd Convergence Analysis')
# plt.plot(baseline, ml, 'o', markersize=5, alpha=0.7, label='Convergence')
plt.scatter(baseline, ml, s=np.array(ds_size)**0.8, alpha=0.7, label='Convergence')
plt.plot([0, 1], [0, 1], 'tab:orange', label='Diagonal')
plt.plot([0, 1], [ml_avg, ml_avg], 'tab:green', label='ML Average')
plt.plot([baseline_avg, baseline_avg], [0, 1], 'tab:red', label='Baseline Average')
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.xlabel('Lloyd + SA Baseline Convergence')
plt.ylabel('ML Convergence')
plt.axis('equal')
plt.grid()
plt.legend()

plt.figure(figsize=(8,8))
plt.title('Convergence of Lloyd and ML vs Problem Size')
plt.semilogx(ds_size, baseline, 'o', label='Baseline Convergence')
plt.semilogx(ds_size, ml, 'o', label='ML Convergence')
plt.xlabel('Problem Size (DOF)')
plt.ylabel('Convergence Rate')
plt.grid()
plt.legend()
plt.show()

plt.figure(figsize=(8,8))
plt.title('Relative performance of ML to Lloyd')
plt.semilogx(ds_size, ml / baseline, 'o', label='Relative Performance')
plt.xlabel('Problem Size (DOF)')
plt.ylabel('Ratio of ML to Lloyd convergence')
plt.grid()
plt.legend()
plt.show()
