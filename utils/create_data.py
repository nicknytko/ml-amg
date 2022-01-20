import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import os
import sys
import argparse
import pygmsh
import scipy.spatial as spat
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
parser.add_argument('--percent-structured', type=float, default=0.3, help='Percent of the created grids that should be on structured meshes')
args = parser.parse_args()

os.makedirs(args.out_folder, exist_ok=True)

# magic coefficients for dof -> mesh size
c = np.array([-5.07631394e-24,  1.18051145e-20, -1.18759608e-17,  6.76116717e-15,
              -2.39110729e-12,  5.41996191e-10, -7.81738597e-08,  6.82384359e-06,
              -3.12626571e-04,  3.62137155e-03,  2.72057000e-01])

for i in range(args.n_grids):
    print(i)
    if i < (args.n_grids * args.percent_structured):
        # Create a random structured grid
        N = np.random.randint(8, 20)
        G = Grid.structured_2d_poisson_dirichlet(N, N)
    else:
        # Create a random convex hull for the boundary, then mesh the interior with gmsh
        N = np.random.randint(50, 250)
        X = np.random.rand(N, 2)
        nv = np.random.randint(25, 400)
        ms = np.polyval(c, nv)

        hull = spat.ConvexHull(X)
        bv = X[hull.vertices]
        with pygmsh.geo.Geometry() as geom:
            geom.add_polygon(bv, ms)
            mesh = geom.generate_mesh()

        G = Grid.meshio_2d_poisson_dirichlet(mesh)
    G.save(os.path.join(args.out_folder, f'{i:04}'))
