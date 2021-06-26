import numpy as np
import numpy.linalg as la
import pyamg
import scipy
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import collections

# reference grid generators

def ref_all_fine(A):
    N = int(A.shape[0])
    z = np.zeros(N)
    return 'All fine', z

def ref_all_coarse(A):
    N = int(A.shape[0])
    z = np.ones(N)
    return 'All coarse', z

def ref_coarsen_by(c):
    def f(A):
        N = int(A.shape[0])
        zz = np.zeros(N)
        for x in range(0,N,c):
            zz[x] = 1
        return f'Coarsening by {c}', zz
    return f

def ref_coarsen_by_bfs(c):
    def f(A):
        N = int(A.shape[0])
        zz = np.zeros(N)

        first = 0
        visited = np.zeros(N, dtype=bool)
        distance = np.zeros(N, dtype=int)

        while np.sum(visited) != N:
            first = np.where(visited == False)[0][0]
            visited[first] = True
            Q = collections.deque([first])

            while len(Q) > 0:
                v = Q.popleft()
                row = A[v].todense()
                adjacent = np.where(row != 0)[1]
                for a in adjacent:
                    if not visited[a]:
                        visited[a] = True
                        distance[a] = distance[v] + 1
                        Q.append(a)

        zz[(distance % c == 0)] = 1
        return f'(BFS) Coarsening by {c}', zz
    return f

def ref_amg(theta=0.25):
    def f(A):
        N = int(A.shape[0])
        S = pyamg.strength.classical_strength_of_connection(A, theta=theta)
        spl = pyamg.classical.RS(S)
        return 'AMG - RS', spl
    return f

# multigrid

def jacobi(A, b, x, omega=0.666, nu=2):
    Dinv = sp.diags(1.0/A.diagonal())
    for i in range(nu):
        x += omega * Dinv @ b - omega * Dinv @ A @ x
    return x

def mg(P, A, b, x, omega=0.666):
    x = jacobi(A, b, x, omega)
    AH = P.T@A@P
    rH = P.T@(b-A@x)
    x += P@spla.spsolve(AH, rH)
    x = jacobi(A, b, x, omega)
    return x

def mgv(P, A, omega=0.666, tol=1e-8):
    err = []
    n = A.shape[0]
    x = np.random.rand(n)
    orig_err = la.norm(x, 2)
    iterations = 0

    while la.norm(x, 2) >= tol:
        x = mg(P, A, np.zeros(n), x, omega)
        iterations += 1
        if iterations >= 500:
            break

    return iterations

def mgavg(P, A, N=20, omega=0.666, tol=1e-8):
    iters = 0
    for i in range(N):
        iters += mgv(P, A, omega, tol) / N
    return int(iters)

def create_interp(A, grid):
    if len(grid.shape) > 1:
        N = grid.shape[0]
        G = grid.reshape(N**2)
    else:
        G = grid
    S = pyamg.strength.classical_strength_of_connection(A, theta=0.25)
    return pyamg.classical.direct_interpolation(A,S,G.astype('intc'))

def det_conv_factor_optimal_omega(P, A, b, x_ref):
    def obj(omega):
        return mgv(P, A, b, x_ref, omega)
    opt = scipy.optimize.minimize_scalar(obj, (0, 1), bounds=(0, 1), method='bounded', options={'maxiter': 50})
    return opt.fun, opt.x
