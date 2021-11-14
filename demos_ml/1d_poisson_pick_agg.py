import torch
import torch.linalg as tla
import torch.nn as nn
import torch_geometric.nn as tgnn
import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import sys
import pyamg
import matplotlib.pyplot as plt
import pygad
import pygad.torchga

sys.path.append('../')
import ns.model.agg_interp
import ns.model.loss
import ns.model.data
import ns.lib.sparse
import ns.lib.sparse_tensor
import ns.lib.multigrid
import ns.lib.graph

neumann_solve = 'neumann' in sys.argv
initial_supervise = 'supervise' in sys.argv

device = 'cpu'
#model = ns.model.agg_interp.FullAggNet(64).to(device)

n_aggs = 3
n_per_agg = 3
n = n_aggs * n_per_agg
A = sp.eye(n) * 2 - sp.eye(n, k=-1) - sp.eye(n, k=1)

# Apply Neumann boundary conditions
if neumann_solve:
    A = A.tolil()
    A[0,0] = 1
    A[0,1] = -1
    A[-1,-1] = 1
    A[-1,-2] = -1

A = A.tocsr()
print(' -- Problem setup --')
print('A\n', A.todense())

A_T = ns.lib.sparse.to_torch_sparse(A).to(device)

alpha = 1. / 3.
omega = 2. / 3.
Dinv = sp.diags([1.0 / A.diagonal()], [0])
smoother = (sp.eye(n) - omega*Dinv @ A)
smoother_T = ns.lib.sparse.to_torch_sparse(smoother).to(device)
A_Graph = ns.model.data.graph_from_matrix_basic(A)

# baseline aggregation
Agg = np.zeros((n, n_aggs))
for agg in range(n_aggs):
    Agg[n_per_agg*agg:n_per_agg*(agg+1), agg] = 1.
Agg = sp.csr_matrix(Agg)
P_SA = smoother @ Agg

def loss_fcn(A, P_T):
    P = ns.lib.sparse_tensor.to_scipy(P_T)
    b = np.zeros(A.shape[1])
    x = np.random.randn(A.shape[1])
    x /= la.norm(x, 2)

    ret = ns.lib.multigrid.amg_2_v(A, P, b, x, res_tol=1e-10, singular=neumann_solve)
    # print(ret)
    return ret[1]

print('Agg (baseline)\n', Agg.todense())
print('Baseline convergence', loss_fcn(A, ns.lib.sparse.to_torch_sparse(P_SA)))

use_network_P = False

def fitness(weights, weights_idx):
    global use_network_P

    model = ns.model.agg_interp.FullAggNet(64)
    model.load_state_dict(pygad.torchga.model_weights_as_dict(model, weights))
    model.eval()

    with torch.no_grad():
        agg, P = model.forward(A, alpha)
    if not use_network_P:
        P = (smoother_T.to_dense() @ agg.to_dense()).to_sparse()
    loss = loss_fcn(A, P)

    if np.isnan(loss.item()):
        return 0
    else:
        return 1.0 / (loss + #amg loss
                      #torch.std(tla.norm(agg.to_dense(), ord=1, dim=0)) * 0.5 + # deviation of aggregate size
                      torch.sum(tla.norm(P.to_dense(), ord=1, dim=1) < 1e-3)) # number of nodes that are not interpolated to

def display_progress(ga_instance):
    global use_network_P

    weights = ga_instance.best_solution()
    if ga_instance.generations_completed >= 20:
        use_network_P = True

    model = ns.model.agg_interp.FullAggNet(64)
    model.load_state_dict(pygad.torchga.model_weights_as_dict(model, weights[0]))
    model.eval()
    with torch.no_grad():
        agg, P = model.forward(A, alpha)
    if not use_network_P:
        P = (smoother_T.to_dense() @ agg.to_dense()).to_sparse()

    print(f'Generation = {ga_instance.generations_completed}')
    print(f'Fitness    = {weights[1]}')
    print(f'Loss       = {1.0/weights[1]}')
    print('Agg\n', agg.to_dense())
    print('P\n', P.to_dense())

if __name__ == '__main__':
    model = ns.model.agg_interp.FullAggNet(64)
    initial_population = pygad.torchga.TorchGA(model=model, num_solutions=50).population_weights
    ga_instance = pygad.GA(num_generations=500,
                           num_parents_mating=5,
                           initial_population=initial_population,
                           fitness_func=fitness,
                           on_generation=display_progress,
                           mutation_probability=0.3,
                           keep_parents=1,
                           num_parallel_workers=None)
    ga_instance.run()
