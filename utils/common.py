import numpy as np
import numpy.linalg as la
import pyamg
import scipy.sparse as sp
import sys
import torch

sys.path.append('../')
import ns.model.agg_interp
import ns.model.data
import ns.lib.sparse
import ns.lib.sparse_tensor
import ns.lib.multigrid
import ns.lib.graph
import ns.ga.parga
import ns.ga.torch

#### Common functions and definitions for utility scripts

strength_measure_funcs = {
    'abs': lambda A: abs(A),
    'evolution': lambda A: pyamg.strength.evolution_strength_of_connection(A) + sp.csr_matrix((np.ones_like(A.data), A.indices, A.indptr), A.shape) * 0.1,
    'invabs': lambda A: sp.csr_matrix((1.0 / np.abs(A.data), A.indices, A.indptr), A.shape),
    'unit': lambda A: sp.csr_matrix((np.ones_like(A.data), A.indices, A.indptr), A.shape),
    'olson': lambda A: pyamg.strength.evolution_strength_of_connection(A) + sp.csr_matrix((1./np.abs(A.data), A.indices, A.indptr), A.shape),
}

def parse_bool_str(v):
    v = v.lower()
    if v == 't' or v == 'true':
        return True
    else:
        return False

def evaluate_dataset(weights, dataset, model=None, S=None, neumann_solve=False, alpha=0.3, omega=2./3.):
    if model is not None:
        model.load_state_dict(ns.ga.torch.model_weights_as_dict(model, weights))
        model.eval()

    conv = np.zeros(len(dataset))
    for i in range(len(dataset)):
        A = dataset[i].A
        if model is not None:
            with torch.no_grad():
                agg_T, P_T, bf_weights, cluster_centers, node_scores = model.forward(A, alpha)
            P = ns.lib.sparse_tensor.to_scipy(P_T)
        else:
            if S is None:
                C = strength_measure_funcs['abs'](A)
            else:
                C = S(A)
            Agg, _ = pyamg.aggregation.lloyd_aggregation(C, ratio=alpha, distance='same')
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

def evaluate_ref_conv(dataset, strength_measure_func, neumann_solve=False, alpha=0.3, omega=2./3.):
    conv = np.zeros(len(dataset))
    for i in range(len(dataset)):
        A = dataset[i].A
        C = strength_measure_func(A)
        Agg, _ = pyamg.aggregation.lloyd_aggregation(C, ratio=alpha, distance='same')
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
