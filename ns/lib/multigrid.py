import numpy as np
import numpy.linalg as la
import pyamg
import scipy
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def jacobi(A, Dinv, b, x, omega=0.666, nu=2):
    for i in range(nu):
        x += omega * Dinv @ b - omega * Dinv @ A @ x
    return x

def amg_2_v(A, P, b, x,
            pre_smoothing_steps=1,
            post_smoothing_steps=1,
            jacobi_weight=0.666,
            res_tol=None,
            error_tol=None,
            max_iter=500):
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
    jacobi_weight : float
        value of omega for Jacobi relaxation scheme
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
    Dinv = sp.diags(1.0/A.diagonal())
    n = A.shape[0]

    while True:
        x = x.copy()

        # Pre-relaxation
        x = jacobi(A, Dinv, b, x, omega=jacobi_weight, nu=pre_smoothing_steps)
        # Coarse-grid correction
        x += P @ spla.spsolve(P.T@A@P, P.T@(b - A@x))
        # Post-relaxation
        x = jacobi(A, Dinv, b, x, omega=jacobi_weight, nu=post_smoothing_steps)
        # Normalize with zero mean for singular systems w/ constant nullspace
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
