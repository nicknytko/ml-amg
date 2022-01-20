import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import os
import sys
import argparse
import pygmsh
import scipy.spatial as spat
import scipy.optimize
sys.path.append('../')

from ns.model.data import Grid

def parse_bool_str(v):
    v = v.lower()
    if v == 't' or v == 'true':
        return True
    else:
        return False

n_grids = 500
mesh_sizes = np.zeros(n_grids, dtype=np.int64)
ms_param = np.zeros(n_grids)

c = np.array([-5.07631394e-24,  1.18051145e-20, -1.18759608e-17,  6.76116717e-15,
              -2.39110729e-12,  5.41996191e-10, -7.81738597e-08,  6.82384359e-06,
              -3.12626571e-04,  3.62137155e-03,  2.72057000e-01])

for i in range(n_grids):
    # Create a random convex hull for the boundary, then mesh the interior with gmsh
    N = np.random.randint(50, 250)
    X = np.random.rand(N, 2)

    hull = spat.ConvexHull(X)
    bv = X[hull.vertices]
    nv = np.random.randint(25, 400)
    p = np.polyval(c, nv)

    ms_param[i] = p
    with pygmsh.geo.Geometry() as geom:
        geom.add_polygon(bv, p)
        mesh = geom.generate_mesh()
    G = Grid.meshio_2d_poisson_dirichlet(mesh)
    mesh_sizes[i] = G.A.shape[0]

mssi = np.argsort(mesh_sizes)
mss = mesh_sizes[mssi]
mps = ms_param[mssi]

deg = 10
c = np.polyfit(mesh_sizes, ms_param, deg)
print(c)

plt.figure()
plt.hist(mesh_sizes)

plt.figure()
plt.scatter(mesh_sizes, ms_param)
plt.plot(mss, np.polyval(c, mss), 'tab:orange')
plt.show(block=True)

for i in range(n_grids):
    # Create a random structured grid
    N = np.random.randint(8, 20)
    G = Grid.structured_2d_poisson_dirichlet(N, N)
    mesh_sizes[i] = G.A.shape[0]

plt.figure()
plt.hist(mesh_sizes)
plt.show(block=True)
