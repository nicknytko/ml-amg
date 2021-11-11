import torch
import torch.linalg as tla
import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import sys
import pyamg
import matplotlib.pyplot as plt

sys.path.append('../')
import ns.model.agg_interp
import ns.model.loss
import ns.model.data
import ns.lib.sparse
import ns.lib.multigrid

# This is a small demo to show we can learn interpolation for aggregation-based AMG
# A small Poisson system is generated using finite differences, then solved for Ax=0

### Arguments:
# pass 'neumann' to solve neumann system
# pass 'supervise' to initially train the network to mimic the Jacobi smoother

neumann_solve = 'neumann' in sys.argv
initial_supervise = 'supervise' in sys.argv

def scipy_to_torch_sparse(A):
    A = A.tocoo()
    indices = torch.vstack([torch.Tensor(A.row), torch.Tensor(A.col)])
    AT = torch.sparse_coo_tensor(indices, torch.Tensor(A.data), A.shape)
    AT = AT.coalesce().float()
    return AT


def torch_sparse_to_scipy(T):
    indices = np.array(T.indices())
    coo =  sp.coo_matrix((np.array(T.values()),
                         (indices[0], indices[1])),
                         shape=np.array(T.shape))
    return coo.tocsr()

device = 'cpu'
model = ns.model.agg_interp.PNet(64, device)

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

A_T = scipy_to_torch_sparse(A).to(device)

omega = 2. / 3.
Agg = np.zeros((n, n_aggs))
for agg in range(n_aggs):
    Agg[n_per_agg*agg:n_per_agg*(agg+1), agg] = 1.
Agg = sp.csr_matrix(Agg)
Agg_T = scipy_to_torch_sparse(Agg).to(device)
print('Agg\n', Agg.todense())

Dinv = sp.diags([1.0 / A.diagonal()], [0])
P_SA = (sp.eye(n) - omega*Dinv @ A) @ Agg
P_SA_T = scipy_to_torch_sparse(P_SA).to(device)
A_Graph = ns.model.data.graph_from_matrix(A, Agg)

lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

if initial_supervise:
    # Supervised loss
    for i in range(200):
        optimizer.zero_grad()
        P = model(A_Graph, Agg)

        diff = (P_SA_T - P).coalesce()
        loss = tla.norm(diff.values())
        print('loss', i+1, ':', loss.item())
        loss.backward()
        optimizer.step()
    print('Finished supervised training')

lr = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# Unsupervised loss
for i in range(200):
    optimizer.zero_grad()
    P = model(A_Graph, Agg)

    test_vecs = torch.tensor(np.random.normal(0, 1, (A.shape[0], 50))).float()
    test_vecs = test_vecs / tla.norm(test_vecs, 2, dim=0)
    test_vecs = test_vecs.to(device)

    loss = ns.model.loss.amg_loss(P, A_T, test_vecs,
                                  tot_num_loop=5, no_prerelax=1,
                                  no_postrelax=1,  device=device,
                                  neumann_solve_fix=neumann_solve)
    loss.backward()
    print('loss', i+1, ':', loss.item())
    optimizer.step()
    if loss.item() < 0.3:
        break

with torch.no_grad():
    P_ml = model(A_Graph, Agg)
    P_hat_ml = model.forward_P_hat(A_Graph)

P_ml = torch_sparse_to_scipy(P_ml)
P_ml_norm = ns.lib.sparse.col_normalize_csr(P_ml, ord=np.inf)
P_hat_ml = torch_sparse_to_scipy(P_hat_ml)

print(' -- Interpolation operators --')
print('SA\n', P_SA.todense())
print('ML\n', P_ml.todense())
print('ML (normalized w/ inf norm)\n', P_ml_norm.todense())
print(r' -- I - \omega D^{-1}A')
print('SA\n', np.round((sp.eye(n) - omega*Dinv @ A).todense(), 3))
print('ML\n', np.round(P_hat_ml.todense(), 3))

x = np.random.normal(0, 1, (A.shape[0],))
b = np.zeros(A.shape[0])
tol = 1e-10
print(' -- Convergence Factors -- ')
print('SA', ns.lib.multigrid.amg_2_v(A, P_SA, b, x, error_tol=tol)[1])
print('ML', ns.lib.multigrid.amg_2_v(A, P_ml, b, x, error_tol=tol)[1])
