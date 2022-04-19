import numpy as np
from firedrake import *
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import numpy.linalg as la
import pyamg.aggregation
import pyamg.strength
import sys

sys.path.append('../')
import ns.model.data

N = 20

mesh = UnitCubeMesh(N,N,N)
V = FunctionSpace(mesh, 'CG', 1)

u = TrialFunction(V)
v = TestFunction(V)

f = Function(V)
x,y,z = SpatialCoordinate(mesh)
f.interpolate(x**0)

theta_z = np.random.uniform(0, 2*np.pi)
theta_y = np.random.uniform(0, 2*np.pi)
#theta_z = 0
#theta_y = 0
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

#eps_x = 10.**np.random.uniform(-4,4)
#eps_y = 10.**np.random.uniform(-4,4)
eps_x = 1e-3
eps_y = 1e3
S = np.diag([eps_x, eps_y, 1.])

print(f'Theta z {theta_z/np.pi:.3f}pi', ';', f'Theta y {theta_y/np.pi:.3f}pi')
print('Epsilon x', eps_x, 'Epsilon y', eps_y)

D = as_matrix(R.T@S@R)
a = inner(D * grad(u), grad(v)) * dx
l = inner(f, v) * dx

# Assemble bilinear form and RHS
xyz = mesh.coordinates.dat.data
A_petsc = assemble(a)
row, col, val = A_petsc.petscmat.getValuesCSR()
A = sp.csr_matrix((val, col, row))
b = (assemble(l).vector().dat.data)

# Acquire boundary mask
boundary_nodes = set()
for i in range(1,7):
    boundary_nodes.update(V.boundary_nodes(i))
interior_nodes = np.array(list(set(range(A.shape[0])) - boundary_nodes))
R_ = (sp.eye(A.shape[0]).tocsr())[interior_nodes]

# Mask off Dirichlet nodes
A = R_@A@R_.T
b = R_@b
n = len(interior_nodes)

# Initial guess
uv = np.random.randn(n)

# Gauss-Seidel relaxation
L = sp.tril(A).tocsr()
U = sp.triu(A, k=1).tocsr()
nu = 10
for i in range(nu):
    uv = spla.spsolve_triangular(L, b-U@uv)

alpha = (.3)**3
Nc = int(np.ceil(alpha*A.shape[0]))
print('Number of DoF', A.shape[0], 'Number of aggregates', Nc)

def olson_strength(A):
    A = A.tocsr()
    ev = pyamg.strength.evolution_strength_of_connection(A)
    inv = sp.csr_matrix((1./np.abs(A.data), A.indices, A.indptr))
    return ev + inv
AggOp, seeds = pyamg.aggregation.balanced_lloyd_aggregation(olson_strength(A), num_clusters=Nc)
agg_assign = np.array(AggOp.argmax(axis=1)).flatten()

colors = np.array(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
agg_colors = colors[agg_assign % len(colors)].tolist()

fig = plt.figure()
ax = plt.axes(projection='3d')
scatter=ax.scatter3D(R_@xyz[:,0], R_@xyz[:,1], R_@xyz[:,2], c=np.abs(uv))
ax.set_title(f'Error after {nu} Gauss-Seidel iterations')
plt.colorbar(scatter, ax=ax)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(R_@xyz[:,0], R_@xyz[:,1], R_@xyz[:,2], s=1.+(np.abs(uv)**0.5)*2e2, c=agg_colors)
ax.set_title('Assignment of nodes to aggregates')

plt.show(block=True)

G = ns.model.data.Grid(A, x=R_@xyz, extra={
    'theta_z': theta_z,
    'theta_y': theta_y,
    'eps_x': eps_x,
    'eps_y': eps_y,
    'eps_z': 1.,
    'dim': 3,
})
G.save('laplace_3d')
