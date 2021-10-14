import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import pickle
import matplotlib.pyplot as plt
import pyamg

def jacobi(A, b, x, omega=0.666, nu=2):
    Dinv = sp.diags(1.0/A.diagonal())
    for i in range(nu):
        x += omega * Dinv @ b - omega * Dinv @ A @ x
    return x

grid_dim=(15,15)
epsilon = 3.0
theta_deg = 45
S = pyamg.gallery.diffusion_stencil_2d(epsilon=epsilon, theta=(theta_deg * (np.pi / 180)))
A = pyamg.gallery.stencil_grid(S, grid=grid_dim, format='csr')

with open('lcurve.pkl', 'rb') as f:
    lcurve = pickle.load(f)

C = lcurve[.0425]['C']
c = C.flatten()

print(C)

P = pyamg.classical.direct_interpolation(A, A, c)

print(P.todense())

n = A.shape[0]
x = np.random.rand(n)
b = np.zeros(n)

x_nrm_history = [la.norm(x, 2)]

for i in range(100):
    x = jacobi(A, b, x, nu=1)
    x += P @ spla.spsolve(P.T@A@P, P.T@(b-A@x))
    x = jacobi(A, b, x, nu=1)
    x_nrm_history.append(la.norm(x, 2))

print((x_nrm_history[-1] / x_nrm_history[-10]) ** (1/9))

plt.figure()
plt.spy(A)
plt.show()

plt.figure()
plt.semilogy(x_nrm_history)
plt.xlabel('iteration count')
plt.ylabel('two norm of error')
plt.title('mg convergence for Ax=0, random initial guess')
plt.show()
