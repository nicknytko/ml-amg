import torch
import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import sys
import os
import pyamg
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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
import utils.common as common

def parse_bool_str(v):
    v = v.lower()
    if v == 't' or v == 'true':
        return True
    else:
        return False

parser = argparse.ArgumentParser(description='Demo of the trained ML-AMG network on a single example')
parser.add_argument('system', type=str, help='Problem selection to run demo on')
parser.add_argument('--n', type=int, default=1000, help='How many different seeds to try')
parser.add_argument('--alpha', type=float, default=None, help='Coarsening ratio for aggregation')
parser.add_argument('--strength-measure', default='olson', choices=common.strength_measure_funcs.keys())
args = parser.parse_args()

N = args.n
alpha = args.alpha
neumann_solve = False

if os.path.exists(args.system):
    if alpha is None:
        alpha = .1
    grid = ns.model.data.Grid.load(args.system)
    A = grid.A
    n = A.shape[0]

    np.random.seed(0)
    C = common.strength_measure_funcs[args.strength_measure](A)
    #Agg, Agg_roots, Agg_seeds = ns.lib.graph.lloyd_aggregation(C, ratio=alpha, distance='same')

    #_, dumb_centers = ns.lib.graph.modified_bellman_ford(ns.lib.sparse.scipy_to_torch(C), torch.Tensor(Agg_seeds).long())
    #Agg_dumb = ns.lib.sparse.torch_to_scipy(ns.lib.graph.nearest_center_to_agg(torch.Tensor(Agg_seeds).long(), dumb_centers))
else:
    print(f'Unknown system {args.system}')
    exit(1)

np.random.seed()

# Set up Jacobi smoother
Dinv = sp.diags([1.0 / A.diagonal()], [0])
omega = (4. / 3.) / np.abs(spla.eigs(Dinv @ A, k=1, return_eigenvectors=False)).item()
smoother = (sp.eye(n) - omega*Dinv@A)

convs = np.zeros(N)
for i in range(N):
    Agg, Agg_roots, Agg_seeds = ns.lib.graph.lloyd_aggregation(C, ratio=alpha, distance='same', rand=i)
    _, dumb_centers = ns.lib.graph.modified_bellman_ford(ns.lib.sparse.scipy_to_torch(C), torch.Tensor(Agg_seeds).long())
    Agg_dumb = ns.lib.sparse.torch_to_scipy(ns.lib.graph.nearest_center_to_agg(torch.Tensor(Agg_seeds).long(), dumb_centers))
    P = smoother@Agg_dumb

    x = np.random.RandomState(0).randn(A.shape[1])
    x /= la.norm(x, 2)

    b = np.zeros(n)
    convs[i] = ns.lib.multigrid.amg_2_v(A, P, b, x, res_tol=1e-6, singular=neumann_solve, jacobi_weight=omega)[1]
    print(i, convs[i])

np.save('lloyd_hist', convs)
