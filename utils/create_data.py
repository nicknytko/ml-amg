import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import os
import sys
import argparse
import pygmsh
import scipy.spatial as spat
import scipy.interpolate as sint
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
parser.add_argument('--percent-structured', type=float, default=0.0, help='Percent of the created grids that should be on structured meshes')
parser.add_argument('--type', choices=['isotropic', 'anisotropic', 'jump', 'noisy', 'smooth'], default='isotropic')
parser.add_argument('--ms', type=float, default=None)
args = parser.parse_args()

os.makedirs(args.out_folder, exist_ok=True)

print(args.out_folder)

# magic coefficients for dof -> mesh size
c = np.array([-5.07631394e-24,  1.18051145e-20, -1.18759608e-17,  6.76116717e-15,
              -2.39110729e-12,  5.41996191e-10, -7.81738597e-08,  6.82384359e-06,
              -3.12626571e-04,  3.62137155e-03,  2.72057000e-01])

for i in range(args.n_grids):
    if i < (args.n_grids * args.percent_structured):
        # Create a random structured grid
        N = np.random.randint(8, 20)
        G = Grid.structured_2d_poisson_dirichlet(N, N)
    else:
        # Create a random convex hull for the boundary, then mesh the interior with gmsh
        N = np.random.randint(50, 250)
        X = np.random.rand(N, 2)
        nv = np.random.randint(25, 400)
        if args.ms is None:
            ms = np.polyval(c, nv)
        else:
            ms = args.ms

        hull = spat.ConvexHull(X)
        bv = X[hull.vertices]
        with pygmsh.geo.Geometry() as geom:
            geom.add_polygon(bv, ms)
            mesh = geom.generate_mesh()

        if args.type == 'isotropic':
            epsilon = 1.0
            theta = 0.0
            G = Grid.meshio_2d_poisson_dirichlet(mesh, epsilon, theta)
        elif args.type == 'anisotropic':
            epsilon = 0
            while abs(epsilon) < 1e-5:
                epsilon = 10 ** np.random.uniform(-5, 5)
                theta = np.random.uniform(0, 2 * np.pi)
            G = Grid.meshio_2d_poisson_dirichlet(mesh, epsilon, theta)
        elif args.type == 'jump':
            Ns = np.random.choice([2, 3])
            S = np.random.uniform(0, 1, (Ns, 2))
            while True:
                D = 10 ** np.random.uniform(-4, 4, Ns)
                if np.ptp(D) > 1e3:
                    break

            J = np.column_stack((S, D))
            G = Grid.meshio_2d_poisson_dirichlet_jump_coeffs(mesh, J)
        elif args.type == 'noisy':
            interior_nodes = set(range(mesh.points.shape[0]))
            for (a, b) in mesh.cells_dict['line']:
                if a in interior_nodes:
                    interior_nodes.remove(a)
                if b in interior_nodes:
                    interior_nodes.remove(b)
            interior_nodes = list(interior_nodes)
            interior_pts = mesh.points[interior_nodes, :2]

            dd = np.random.uniform(0., 4., size=interior_pts.shape[0])
            ddint = sint.NearestNDInterpolator(interior_pts, dd)

            def diffusion_kappa(x,y):
                return 10.**ddint(x,y)
            G = Grid.meshio_2d_poisson_dirichlet_custom(mesh, diffusion_kappa, {'noise': dd})
        elif args.type == 'smooth':
            interior_nodes = set(range(mesh.points.shape[0]))
            for (a, b) in mesh.cells_dict['line']:
                if a in interior_nodes:
                    interior_nodes.remove(a)
                if b in interior_nodes:
                    interior_nodes.remove(b)
            interior_nodes = list(interior_nodes)
            interior_pts = mesh.points[interior_nodes, :2]

            # parameters
            theta = np.random.uniform(0, 2*np.pi)
            xs = np.random.uniform(0.1, 10)
            ys = np.random.uniform(0.1, 10)
            R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            D = np.diag([xs,ys])
            b = np.random.uniform(-10, 10, size=(2,))

            D_S = R.T @ (D @ R @ interior_pts.T + b.reshape((2,1)))
            dd = (np.cos(D_S[0]) ** 2 + np.cos(D_S[1]) ** 2)*1.5 + 0.2
            ddint = sint.NearestNDInterpolator(interior_pts, dd)

            def diffusion_kappa(x,y):
                return 10.**ddint(x,y)
            G = Grid.meshio_2d_poisson_dirichlet_custom(mesh, diffusion_kappa, {'noise': dd})

    print(i, G.A.shape[0])
    G.save(os.path.join(args.out_folder, f'{i:04}'))
