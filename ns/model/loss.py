import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torch.linalg as tla
import torch.sparse
import torch_sparse_solve as tss
import torch_sparse as ts
import ns.lib.sparse_tensor as spt

def add_lagrange_rowcols(A, device='cpu'):
    A = A.coalesce()
    i = A.indices()
    v = A.values()
    n, m = A.size()

    new_indices = torch.zeros((2, n+m))
    new_indices[0, :n] = torch.arange(n)
    new_indices[1, :n] = m
    new_indices[0, n:] = n
    new_indices[1, n:] = torch.arange(m)

    new_values = torch.ones(n+m)

    return torch.sparse_coo_tensor(torch.cat((i, new_indices.to(device)), dim=1),
                                   torch.cat((v, new_values.to(device))),
                                   (n+1, m+1)).coalesce()

def add_lagrange_vec(x, device='cpu'):
    return torch.cat((x, torch.zeros(1, x.shape[1]).to(device)), dim=0)

def amg_loss(P, A, test_vecs, tot_num_loop=5, no_prerelax=1,
             no_postrelax=1, device='cpu', neumann_solve_fix=False):
    '''
    Compute the approximate AMG loss of an interpolation operator by running
    the AMG iterations.

    P - sparse torch tensor, N_F x N_C interpolation operator
    A - sparse torch tensor, N_F x N_C system
    test_vecs - {int, torch tensor}, test vectors to use.  If int, will randomly generate them.
    tot_num_loop - int, number of AMG iterations to run.
    no_prerelax - int, number of pre-relaxation iterations to run.
    no_postrelax - int, number of post-relaxation iterations to run.
    device - string, torch device that everything will be running on.  Inputs are expected to be on this device.
    neumann_solve_fix - bool, Does this A have a constant nullspace that we should be handling?
    '''

    omega = 2. / 3.
    D = spt.diag(A)
    Dinv_v = (1. / D) * omega
    Dinv = torch.diag(Dinv_v).to_sparse().coalesce().to(device)

    Pt = spt.spT(P)
    A_H = spt.spspmm(spt.spspmm(Pt, A), P).double()
    N = A.shape[0]

    if not isinstance(test_vecs, torch.Tensor):
        np.random.seed(0)
        x = torch.tensor(np.random.normal(0, 1, (N, test_vecs))).float() # Generating test vectors
        x = x / tla.norm(x, 2, dim=0)
        x = x.to(device)
    else:
        x = test_vecs

    errs = torch.zeros((tot_num_loop+1, x.shape[1])).to(device)
    if neumann_solve_fix:
        A_H = add_lagrange_rowcols(A_H, device)
    #A_H_pinv = tla.pinv(A_H.to_dense())

    for no_loop in range(tot_num_loop+1):
        for j in range(no_prerelax):
            x = x - Dinv @ (A @ x)

        ## solve for the coarse-grid correction
        r_H = spt.spmm(Pt, spt.spmm(A, x))
        if neumann_solve_fix:
            r_H = add_lagrange_vec(r_H, device)

        e_H = tss.solve(A_H.unsqueeze(0), (-r_H).unsqueeze(0).double()).squeeze(0).float()
        #e_H = (A_H_pinv @ -r_H.double()).float()
        if neumann_solve_fix:
            e_H = e_H[:-1]

        ## interpolate coarse error to fine grid
        x = x + spt.spmm(P, e_H)

        for j in range(no_postrelax):
            x = x - Dinv @ (A @ x)
        x = x - x.mean(0)
        errs[no_loop] = tla.vector_norm(x, ord=2, dim=0)

    n_err = 3
    convs = (errs[-1] / errs[-n_err]) ** (1/(n_err-1))
    loss = nnF.softmax(convs, dim=0) @ convs

    return loss
