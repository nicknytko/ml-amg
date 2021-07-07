import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.io as sio
import pyamg

mats = sio.loadmat('firedrake_0_fieldsplit_0_assembled_pyamg_mat_0001.mat')
A = mats['A']
A_amg = pyamg.classical.ruge_stuben_solver(A.tocsr())

x_ref = np.random.rand(A.shape[0])

b = A@x_ref

print('How "asymmetric" the matrix is:', spla.norm(A.T-A))

x_lu = spla.spsolve(A, b)
print(' -- scipy sparse solve -- ')
print('error', la.norm(x_ref-x_lu))
print('residual', la.norm(b-A@x_lu))
print()

x_gmres = spla.gmres(A, b)[0]
print(' -- scipy gmres -- ')
print('error', la.norm(x_ref-x_gmres))
print('residual', la.norm(b-A@x_gmres))
print()

x_amg = A_amg.solve(b, tol=1e-12, accel="gmres")
print(' -- pyamg ruge-stuben solver -- ')
print(A_amg)
print('error', la.norm(x_ref-x_amg))
print('residual', la.norm(b-A@x_amg))
