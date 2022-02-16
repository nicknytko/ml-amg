import numpy as np
import numpy.linalg as la
import pyamg
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import warnings

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

# def sor(A, b, x, L=None, U=None, omega=1., nu=2):
#     '''
#     Successive over-relaxation iterative solver

#     Parameters
#     ----------
#     A : numpy.ndarray, scipy.sparse.spmatrix
#         n x n linear system to solve
#     L : numpy.ndarray, scipy.sparse.spmatrix
#         Lower triangular half of A
#     U : numpy.ndarray, scipy.sparse.spmatrix
#         Strict upper triangular half of A
#     b : numpy.ndarray
#         Length n right hand side to solve for
#     x : numpy.ndarray
#         Length n initial guess for solution
#     omega : float
#         Relaxation weight.  Omega > 1 over-relaxes, while omega < 1 under-relaxes.
#     nu : integer
#         Number of sweeps to perform

#     Returns
#     -------
#     x : numpy.ndarray
#         Length n approximation to solution
#     '''

#     if L is None:
#         L = sp.tril(A)
#     if U is None:
#         U = sp.triu(A, k=1)

#     for i in range(nu):
#         x_gs = gauss_Seidel(A, b, x, L, U, nu=1)
#         x = omega * x_gs + (1-omega) * x
#     return x


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

    try:
        err_n = min(len(err) // 3, 10)
        conv_factor = (err[-1] / err[-err_n]) ** (1/(err_n - 1))
    except:
        conv_factor = 0 # Divide by zero... assume it converged too quickly?

    return x, conv_factor, err, len(err)
