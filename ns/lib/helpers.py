import numpy as np
import numpy.linalg as la
import scipy
import scipy.optimize
import scipy.linalg as sla
import scipy.io as sio
import scipy.sparse as sp
import matplotlib.pyplot as plt
import torch
import pickle
import subprocess
import bz2

def pickle_load(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

def pickle_load_bz2(fname):
    with bz2.BZ2File(fname, 'rb') as f:
        return pickle.load(f)

def pickle_save(fname, obj):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)

def pickle_save_bz2(fname, obj):
    with bz2.BZ2File(fname, 'wb') as f:
        pickle.dump(obj, f)

def grid_to_pytorch(grid):
    g = np.zeros_like(grid, dtype=np.float64)
    for i, x in enumerate(grid):
        g[i] = (1.0 if x else -1.0)
    return g

def grid_to_tensor(grid):
    T = torch.Tensor(grid_to_pytorch(grid))
    return T.reshape((1, 1, T.shape[0]))

def ideal_interpolation(A, picked_C):
    """
    Constructs an ideal interpolation operator.  Shamelessly stolen from Luke Olson's code.

    A - matrix system
    picked_C - boolean numpy array of size 'n' containing 'True' for coarse points and 'False' for fine points.
    returns prolongation matrix P, such that P=R^T
    """
    C = np.where(picked_C)[0]
    F = np.where(np.logical_not(picked_C))[0]
    n = len(picked_C)

    AFF = A[F,:][:,F]
    AFC = A[F,:][:,C]
    ACF = A[C,:][:,F]

    P = np.zeros((n, len(C)))
    P[C,:] = np.eye(len(C))
    P[F,:] = -np.linalg.inv(AFF) @ AFC

    return P

def display_grid(tf):
    plt.figure(figsize=(10,3))
    xs = np.linspace(-1, 1, len(tf))
    ys = np.zeros(len(tf))
    C = np.where(tf)
    F = np.where(np.logical_not(tf))
    plt.plot(xs[C], ys[C], 'rs', ms=15, markerfacecolor="None", markeredgecolor='red', markeredgewidth=2, label="C Pts")
    plt.plot(xs[F], ys[F], 'bo', ms=15, markerfacecolor="None", markeredgecolor='blue', markeredgewidth=2, label="F Pts")
    plt.legend()

def display_grid_2d(G):
    plt.figure(figsize=(10,10))

    n, m = G.shape
    x = np.linspace(0, 1, m)
    y = np.linspace(0, 1, n)

    xx, yy = np.meshgrid(x, y)

    C = np.where(G > 0)
    F = np.where(G <= 0)
    plt.plot(xx[C], yy[C], 'rs', ms=15, markerfacecolor="None", markeredgecolor='red', markeredgewidth=2, label="C Pts")
    plt.plot(xx[F], yy[F], 'bo', ms=15, markerfacecolor="None", markeredgecolor='blue', markeredgewidth=2, label="F Pts")
    plt.legend()

def display_many_grids(T):
    n = T.shape[0]
    plt.figure(figsize=(10,3*n))

    for i, tf in enumerate(T):
        xs = np.linspace(-1, 1, len(tf))
        ys = np.ones(len(tf)) * i
        C = np.where(tf)
        F = np.where(np.logical_not(tf))
        plt.plot(xs[C], ys[C], 'rs', ms=5, markerfacecolor="None", markeredgecolor='red', markeredgewidth=2, label="C Pts")
        plt.plot(xs[F], ys[F], 'bo', ms=5, markerfacecolor="None", markeredgecolor='blue', markeredgewidth=2, label="F Pts")

def relax(A, u0, f, nu=1, omega=0.666):
    u = u0.copy()
    n = A.shape[0]
    Dinv = np.diag(1.0 / np.diag(A))
    for steps in range(nu):
        u += omega * Dinv @ (f - A @ u)
    return u

def twolevel(A, P, A1, u0, f0, nu=1, omega=0.666):
    u0 = relax(A, u0, f0, nu, omega) # pre-smooth
    f1 = P.T @ (f0 - A @ u0)  # restrict

    u1 = la.solve(A1, f1)  # coarse solve

    u0 = u0 + P @ u1          # interpolate
    u0 = relax(A, u0, f0, nu, omega) # post-smooth
    return u0

def disp_grid_convergence(A, x, picked_C, u, omega=0.666):
    """
    Plot a C/F grid and the error dissipation after a few iterations.

    A - system of equations
    x - rhs
    picked_C - boolean numpy array of size 'n' containing 'True' for coarse points and 'False' for fine points
    u - initial guess vector of size 'n'
    omega - Jacobi weight
    """

    P = ideal_interpolation(A, picked_C)
    u = u.copy()

    res_array = []
    e_array = []

    A1 = P.T @ A @ P

    display_grid(picked_C)
    u_ref = la.solve(A, x)

    N = A.shape[0]

    for i in range(15):
        u = twolevel(A, P, A1, u, x, 5, omega)
        res = A@u - x
        e = u - u_ref
        plt.plot(np.linspace(-1,1,N), e)
        res_array.append(res)
        e_array.append(e)

    res_array = np.array(res_array)
    e_array = np.array(e_array)

    conv_factor = np.mean(la.norm(e_array[1:], axis=1) / la.norm(e_array[:-1], axis=1))
    return conv_factor

def grid_from_coarsening_factor(n, f):
    if f > 1:
        f = int(f)
        C = np.array([False]*n)
        for i in range((n-1)%f // 2, n, f):
            C[i] = True
        return C, np.logical_not(C)
    else:
        F = np.array([False]*n)
        f = int(1/f)
        for i in range((n-1)%f // 2, n, f):
            F[i] = True
        return np.logical_not(F), F

def det_conv_fact(A, picked_C, x, u, u_ref, omega):
    P = ideal_interpolation(A, picked_C)
    u = u.copy()

    res_array = []
    e_array = []

    A1 = P.T @ A @ P

    for i in range(15):
        u = twolevel(A, P, A1, u, x, 1, omega)
        res = A@u - x
        e = u - u_ref
        res_array.append(res)
        e_array.append(e)

    res_array = np.array(res_array)
    e_array = np.array(e_array)

    conv_factor = np.mean(la.norm(e_array[1:], axis=1) / la.norm(e_array[:-1], axis=1))
    return conv_factor

def det_conv_factor_optimal_omega(A, picked_C, x, u, u_ref):
    omega_trials = np.linspace(0.01, 0.99, 100)
    conv = 1
    best_omega = 0

    for omega in omega_trials:
        cur_conv = det_conv_fact(A, picked_C, x, u, u_ref, omega)
        if cur_conv < conv:
            conv = cur_conv
            best_omega = omega

    return conv, best_omega

def det_conv_factor_optimal_omega_numopt(A, picked_C, x, u, u_ref):
    P = ideal_interpolation(A, picked_C)
    A1 = P.T @ A @ P

    def obj(omega):
        u = np.zeros(A.shape[0])

        I = 15
        e_array = np.zeros(I)

        for i in range(I):
            u = twolevel(A, P, A1, u, x, 1, omega)
            res = A@u - x
            e = u - u_ref
            e_array[i] = la.norm(e)

        conv_factor = np.mean(e_array[1:] / e_array[:-1])
        return conv_factor

    opt = scipy.optimize.minimize_scalar(obj, (0, 1), bounds=(0, 1), method='bounded', options={'maxiter': 50})
    return opt.fun, opt.x

def random_u(n, scale=1):
    return (2 * (np.random.rand(n) - 0.5)) * scale

def random_grid(n):
    return np.random.choice([True, False], size=n, replace=True)

def gen_1d_poisson_fd(N):
    h = (1.0 / (N + 1))
    A = (1.0/h**2) * (np.eye(N) * 2 - (np.eye(N, k=-1) + np.eye(N, k=1)))
    return A

def gen_1d_poisson_fd_vc(N, k):
    assert(len(k) == N+1)
    h = 1.0/(N+1)
    offdiag = k[1:-1]
    maindiag = k[1:] + k[:-1]
    A = (1.0/h**2) * ( -1 * np.diag(offdiag, k=-1) + -1 * np.diag(offdiag,k=1) + np.diag(maindiag) )
    return A

def midpt(x):
    return np.average(np.column_stack([x[1:], x[:-1]]), axis=1)

def load_recirc_flow(fname='recirc-flow-25.mat'):
    loaded = sio.loadmat(fname)
    A = sp.csr_matrix(loaded['A'])
    b = np.array(loaded['b']).flatten()
    return A,b

def get_root_dir():
    return subprocess.run(['git', 'rev-parse', '--show-toplevel'], capture_output=True).stdout.decode('utf-8').strip()

def get_sparse_mat_indices():
    matrices = list(filter(lambda m: m[0] != '.', sorted(os.listdir(get_root_dir()))))
    return dict(zip(range(len(matrices)), matrices))

def load_sparse_mat(idx):
    indices = get_sparse_mat_indices()
    return sio.loadmat(indices[idx])[0][0][2]

def normalize_mat(A):
    """
    Normalizes all entries of the sparse matrix A such that all nonzero entries are between 0 and 1.
    """

    And = np.abs(A.data.copy())
    And -= np.min(And)
    And /= np.max(And)
    And = (And * 0.9) + 0.1
    return sp.csr_matrix((And, A.indices, A.indptr), shape=A.shape)

def ary_swap(ary, a, b):
    t = ary[a]
    ary[a] = ary[b]
    ary[b] = t

def quick_partition(ary, left, right, pivot):
    v = ary[pivot]

    # move pivot to end
    ary_swap(ary, pivot, right)

    store_idx = left
    for i in range(left, right):
        if ary[i] > v:
            ary_swap(ary, store_idx, i)
            store_idx += 1

    # move pivot to final place
    ary_swap(ary, store_idx, right)

    return store_idx

def quick_select(ary, left, right, k):
    if left == right:
        return ary[left]

    pivot_idx = np.random.randint(left, right+1)
    pivot_idx = quick_partition(ary, left, right, pivot_idx)

    if k == pivot_idx:
        return ary[k]
    elif k < pivot_idx:
        return quick_select(ary, left, pivot_idx - 1, k)
    else:
        return quick_select(ary, pivot_idx + 1, right, k)

def topk_vec(x, k):
    out = None

    if isinstance(x, torch.Tensor):
        out = torch.zeros_like(x)
        x = x.detach().cpu().numpy().flatten()
    elif isinstance(x, list):
        x = np.array(x)
    elif isinstance(x, np.ndarray):
        out = np.zeros_like(x)

    if len(x.shape) != 1:
        x = x.flatten()
    assert(len(x.shape) == 1)

    k_smallest = quick_select(np.copy(x), 0, len(x)-1, k-1)
    out[x>=k_smallest] = 1.0
    return out
