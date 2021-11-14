import torch
import torch.linalg as tla
import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import sys
import pyamg
import matplotlib.pyplot as plt
import nevergrad as ng

sys.path.append('../')
import ns.model.agg_interp
import ns.model.loss
import ns.model.data
import ns.lib.sparse
import ns.lib.sparse_tensor
import ns.lib.multigrid

neumann_solve = 'neumann' in sys.argv
device = 'cpu'

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

omega = 2. / 3.
Agg = np.zeros((n, n_aggs))
for agg in range(n_aggs):
    Agg[n_per_agg*agg:n_per_agg*(agg+1), agg] = 1.
Agg = sp.csr_matrix(Agg)
Agg_T = ns.lib.sparse.to_torch_sparse(Agg).to(device)
print('Agg\n', Agg.todense())

Dinv = sp.diags([1.0 / A.diagonal()], [0])
P_SA = (sp.eye(n) - omega*Dinv @ A) @ Agg
P_SA_T = ns.lib.sparse.to_torch_sparse(P_SA).to(device)
A_Graph = ns.model.data.graph_from_matrix(A, Agg)

# Define \hat{P}, which is what we will be optimizing the nonzero entries of

def training_loop_phat(lr: float, dims: int):
    P_hat = ns.lib.sparse.to_torch_sparse(sp.eye(n) - omega*Dinv @ A)
    P_hat_indices = P_hat.indices()
    P_hat_vals = torch.ones_like(P_hat.values())
    P_hat_shape = P_hat.size()
    P_hat_vals.requires_grad = True

    optimizer = torch.optim.Adam([P_hat_vals], lr=lr)
    for i in range(200):
        optimizer.zero_grad()
        P_hat = torch.sparse_coo_tensor(P_hat_indices, P_hat_vals, P_hat_shape).coalesce()
        P = torch.sparse.mm(P_hat, Agg_T).coalesce()

        np.random.seed(0)
        test_vecs = torch.tensor(np.random.normal(0, 1, (A.shape[0], 50))).float()
        test_vecs = test_vecs / tla.norm(test_vecs, 2, dim=0)
        test_vecs = test_vecs.to(device)

        loss = ns.model.loss.amg_loss(P, A_T, test_vecs,
                                      tot_num_loop=20, no_prerelax=1,
                                      no_postrelax=1,  device=device,
                                      neumann_solve_fix=neumann_solve)
        loss.backward()
        optimizer.step()
        if loss.item() < 0.3:
            break

    print(f'lr={lr}, dims={dims}: {loss.item()}')
    return loss.item(), P, P_hat

def training_loop(lr: float, dims: int):
    model = ns.model.agg_interp.PNet(dims, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for i in range(200):
        optimizer.zero_grad()
        P = model(A_Graph, Agg)

        np.random.seed(0)
        test_vecs = torch.tensor(np.random.normal(0, 1, (A.shape[0], 50))).float()
        test_vecs = test_vecs / tla.norm(test_vecs, 2, dim=0)
        test_vecs = test_vecs.to(device)

        loss = ns.model.loss.amg_loss(P, A_T, test_vecs,
                                      tot_num_loop=20, no_prerelax=1,
                                      no_postrelax=1,  device=device,
                                      neumann_solve_fix=neumann_solve)
        loss.backward()
        optimizer.step()
        if loss.item() < 0.3:
            break

    print(f'lr={lr}, dims={dims}: {loss.item()}')
    return loss.item(), P, P_hat

def loss_shim(fcn):
    def f(lr, dims):
        return fcn(lr, dims)[0]
    return f

#print(
conv, P, P_hat = training_loop_phat(0.04815052448749936, 38)
print(conv, '\nP\n', P.to_dense(), '\nP_hat\n', P_hat.to_dense())

param = ng.p.Instrumentation(
    lr=ng.p.Log(lower=1e-4, upper=1.0),
    dims=ng.p.Scalar(lower=1, upper=128).set_integer_casting()
)
meta_opt = ng.optimizers.NGOpt(parametrization=param, budget=200)
recommendation = meta_opt.minimize(loss_shim(training_loop_phat))
print(recommendation)

# P_ML = ns.lib.sparse_tensor.to_scipy(P.detach())
# P_hat_ML = ns.lib.sparse_tensor.to_scipy(P_hat.detach())

# print(' -- Interpolation operators --')
# print('SA\n', P_SA.todense())
# print('ML\n', P_ML.todense())
# print(r' -- I - \omega D^{-1}A')
# print('SA\n', np.round((sp.eye(n) - omega*Dinv @ A).todense(), 3))
# print('ML\n', np.round(P_hat_ML.todense(), 3))

# x = np.random.normal(0, 1, (A.shape[0],))
# b = np.zeros(A.shape[0])
# tol = 1e-10
# print(' -- Convergence Factors -- ')
# print('SA', ns.lib.multigrid.amg_2_v(A, P_SA, b, x, error_tol=tol, singular=neumann_solve)[1])
# print('ML', ns.lib.multigrid.amg_2_v(A, P_ML, b, x, error_tol=tol, singular=neumann_solve)[1])
