import numpy as np
from firedrake import *
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import numpy.linalg as la
import pyamg.aggregation
import pyamg.strength
import sys
import argparse
import os

sys.path.append('../')
from ns.model.data import Grid

def parse_bool_str(v):
    v = v.lower()
    if v == 't' or v == 'true':
        return True
    else:
        return False

parser = argparse.ArgumentParser(description='Creation of diffusion data files')
parser.add_argument('--n-grids', type=int, default=1000, help='Number of grids to generate')
parser.add_argument('--out-folder', type=str, default=None, help='Output directory to put grids')
args = parser.parse_args()

os.makedirs(args.out_folder, exist_ok=True)
print(args.out_folder)


def gen_aniso_laplace(Nx, Ny, Nz, theta_y, theta_z, eps_x, eps_y):
    mesh = UnitCubeMesh(Nx, Ny, Nz)
    V = FunctionSpace(mesh, 'CG', 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    f = Function(V)
    x,y,z = SpatialCoordinate(mesh)
    f.interpolate(x**0)

    R_z = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z),  np.cos(theta_z), 0],
        [              0,                0, 1]
    ])
    R_y = np.array([
        [ np.cos(theta_y), 0, np.sin(theta_y)],
        [               0, 1,               0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
    ])
    R = R_y@R_z
    S = np.diag([eps_x, eps_y, 1.])
    D = as_matrix(R.T@S@R)

    a = inner(D * grad(u), grad(v)) * dx
    l = inner(f, v) * dx

    # assemble bilinear form and RHS
    xyz = mesh.coordinates.dat.data
    A_petsc = assemble(a)
    row, col, val = A_petsc.petscmat.getValuesCSR()
    A = sp.csr_matrix((val, col, row))

    # acquire boundary mask
    boundary_nodes = set()
    for i in range(1, 7): # (Number of sides to box)
        boundary_nodes.update(V.boundary_nodes(i))
    interior_nodes = np.array(list(set(range(A.shape[0])) - boundary_nodes))
    R_ = (sp.eye(A.shape[0]).tocsr())[interior_nodes]

    # mask off dirichlet nodes
    return (R_@A@R_.T).tocsr(), R_@xyz


for i in range(args.n_grids):
    N = np.random.randint(8, 15)
    theta_z = np.random.uniform(0, 2*np.pi)
    theta_y = np.random.uniform(0, 2*np.pi)
    eps_x = 10. ** np.clip(np.random.normal(0., 3.), -4., 4.)
    eps_y = 10. ** np.clip(np.random.normal(0., 3.), -4., 4.)

    A, x = gen_aniso_laplace(N, N, N, theta_y, theta_z, eps_x, eps_y)
    G = Grid(A, x=x, extra={
        'theta_z': theta_z,
        'theta_y': theta_y,
        'eps_x': eps_x,
        'eps_y': eps_y,
        'eps_z': 1.,
        'dim': 3,
    })
    print(i, G.A.shape[0])
    G.save(os.path.join(args.out_folder, f'{i:04}'))
