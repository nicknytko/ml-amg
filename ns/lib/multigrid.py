import numpy as np
import numpy.linalg as la
import pyamg
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import ns.lib.sparse
import warnings

import torch
import torch.linalg as tla

def jacobi(A, b, x, Dinv=None, omega=0.666, nu=2):
    '''
    Weighted Jacobi iterative solver

    Parameters
    ----------
    A : numpy.ndarray, scipy.sparse.spmatrix
        n x n linear system to solve
    Dinv : numpy.ndarray, scipy.sparse.spmatrix
        Inverse to diagonal of A
    b : numpy.ndarray
        Length n right hand side to solve for
    x : numpy.ndarray
        Length n initial guess for solution
    omega : float
        Weighting value for Jacobi sweeps
    nu : integer
        Number of sweeps to perform

    Returns
    -------
    x : numpy.ndarray
        Length n approximation to solution
    '''

    if Dinv is None:
        Dinv = sp.diags(1.0/A.diagonal())

    for i in range(nu):
        x += omega * Dinv @ b - omega * Dinv @ A @ x
    return x


def jacobi_torch(A, b, x, Dinv=None, omega=0.666, nu=2):
    if Dinv is None:
        Dinv = ns.lib.sparse.get_diagonal(A)

    for i in range(nu):
        x += omega * (Dinv * b) - omega * Dinv * (A @ x)

    return x


def gauss_seidel(A, b, x, L=None, U=None, nu=2):
    '''
    Gauss-Seidel iterative solver

    Parameters
    ----------
    A : numpy.ndarray, scipy.sparse.spmatrix
        n x n linear system to solve
    L : numpy.ndarray, scipy.sparse.spmatrix
        Lower triangular half of A
    U : numpy.ndarray, scipy.sparse.spmatrix
        Strict upper triangular half of A
    b : numpy.ndarray
        Length n right hand side to solve for
    x : numpy.ndarray
        Length n initial guess for solution
    nu : integer
        Number of sweeps to perform

    Returns
    -------
    x : numpy.ndarray
        Length n approximation to solution
    '''

    if L is None:
        L = sp.tril(A).tocsr()
    if U is None:
        U = sp.triu(A, k=1).tocsr()

    for i in range(nu):
        x = spla.spsolve_triangular(L, (b - U@x))
    return x


def gauss_seidel_torch(A, b, x, L=None, U=None, nu=2):
    if U is None:
        U = ns.lib.sparse.triu(A, 1)

    for i in range(nu):
        x = tla.solve_triangular(A, torch.unsqueeze(b - U@x, 0), upper=False).squeeze()
    return x


def smoothed_aggregation_jacobi(A, Agg):
    n = A.shape[0]
    Dinv = sp.diags([1.0 / A.diagonal()], [0])
    omega = (4. / 3.) / np.abs(spla.eigs(Dinv @ A, k=1, return_eigenvectors=False)).item()
    smoother = (sp.eye(n) - omega*Dinv@A)
    P = smoother @ Agg
    return P


def amg_2_v(A, P, b, x,
            pre_smoothing_steps=1,
            post_smoothing_steps=1,
            jacobi_weight=0.666,
            res_tol=None,
            error_tol=None,
            max_iter=500,
            singular=False):
    '''
    Two-level AMG solver.

    Parameters
    ----------
    A : array, matrix, sparse matrix
        n x n linear system to solve
    P : array, matrix, sparse matrix
        n_F x n_C interpolation matrix
    b : array
        rhs for linear system, should be an array of shape (n,)
    x : array
        initial guess for solution, should be an array of shape (n,)
    jacobi_weight : (deprecated) float
        value of omega for Jacobi relaxation scheme.  No longer necessary because we are using Gauss-Seidel.
    res_tol : None, float
        if set, will stop iteration when absolute solution residual is below this value
    error_tol : None, float
        if set, will stop iteration when absolute solution norm is below this value
    max_iter : int
        maximum number of iterations before algorithm is stopped

    Returns
    -------
    (x, conv_factor, err, num_iterations)
    x : array
        approximate solution to the system
    conv_factor : float
        convergence factor, approximate to how much error is "dissipated" at each iteration
    err : array
        history of residuals or errors, depending on which tolerance is set
    num_iterations : int
        number of iterations completed until convergence
    '''


    if res_tol is None and error_tol is None:
        raise RuntimeError('One of res_tol or error_tol must be set!')
    else:
        tol = res_tol if res_tol is not None else error_tol

    err = []
    L = sp.tril(A).tocsr()
    U = sp.triu(A, k=1).tocsr()
    n = A.shape[0]

    A_H = P.T@A@P
    if not singular:
        try:
            A_H_LU = spla.factorized(A_H)
        except:
            return x, np.float64(1.), err, 0

    while True:
        x = x.copy()

        # Pre-relaxation
        x = gauss_seidel(A, b, x, L, U, nu=pre_smoothing_steps)
        # Coarse-grid correction
        if singular:
            x += P @ spla.lsqr(P.T@A@P, P.T@(b - A@x))[0]
        else:
            x += P @ A_H_LU(P.T@(b - A@x))

        # Post-relaxation
        x = gauss_seidel(A, b, x, L, U, nu=post_smoothing_steps)
        # Normalize with zero mean for singular systems w/ constant nullspace
        if singular:
            x -= np.mean(x)

        # Compute error/residual norm
        if res_tol is not None:
            e = la.norm(b - A@x, 2)
        else:
            e = la.norm(x, 2)

        # tol check
        err.append(e)
        if e <= tol:
            break
        if len(err) >= max_iter:
            break

    if len(err) != 1:
        try:
            err_n = min(len(err) // 3, 10)
            conv_factor = (err[-1] / err[-err_n]) ** (1/(err_n - 1))
        except:
            conv_factor = 0 # Divide by zero... assume it converged too quickly?
    else:
        conv_factor = 0

    return x, conv_factor, err, len(err)


def amg_2_v_torch(A, P, b, x,
            pre_smoothing_steps=1,
            post_smoothing_steps=1,
            jacobi_weight=0.666,
            error_tol=1e-10,
            max_iter=20):
    device = A.device
    Dinv = 1./ns.lib.sparse.get_diagonal(A)
    # U = ns.lib.sparse.triu(A, 1)
    Pt = P.transpose(0, 1)
    A_H = torch.sparse.mm(torch.sparse.mm(Pt, A), P).to_dense()
    A_H_LU = torch.lu(A_H)

    err = torch.zeros(max_iter).to(device)

    for i in range(max_iter):
        # pre-relaxation
        x = jacobi_torch(A, b, x, Dinv, omega=jacobi_weight, nu=pre_smoothing_steps)
        # x = gauss_seidel_torch(A, b, x, U, nu=pre_smoothing_steps)

        # coarse-grid correction
        r_H = torch.unsqueeze(Pt.matmul(b - A@x), 1)
        e_H = torch.lu_solve(r_H, *A_H_LU)
        x += P.matmul(e_H.squeeze())

        # post-relaxation
        x = jacobi_torch(A, b, x, Dinv, omega=jacobi_weight, nu=post_smoothing_steps)
        # x = gauss_seidel_torch(A, b, x, U, nu=post_smoothing_steps)

        # tolerance check
        err[i] = tla.norm(x)
        if err[i] < error_tol:
            break

    n_err = 2
    return (err[i] / err[i-n_err]) ** (1/(n_err - 1))
