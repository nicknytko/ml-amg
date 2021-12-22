import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import os
import sys
import argparse
sys.path.append('../')

from ns.model.data import Grid

def parse_bool_str(v):
    v = v.lower()
    if v == 't' or v == 'true':
        return True
    else:
        return False

parser = argparse.ArgumentParser(description='Creation of diffusion data files')
parser.add_argument('--n-min', type=int, default=5**2, help='Minimum size of the system')
parser.add_argument('--n-max', type=int, default=15**2, help='Maximum size of the system')
parser.add_argument('--n-grids', type=int, default=1000, help='Number of grids to generate')
parser.add_argument('--out-folder', type=str, default=None, help='Output directory to put grids')
parser.add_argument('--percent-structured', type=float, default=0.3, help='Percent of the created grids that should be on structured meshes')
args = parser.parse_args()

os.makedirs(args.out_folder, exist_ok=True)

for i in range(args.n_grids):
    print(i)
    if i < (args.n_grids * args.percent_structured):
        N = np.random.randint(np.ceil(args.n_min ** 0.5), np.floor((args.n_max+1) ** 0.5))
        G = Grid.structured_2d_poisson_dirichlet(N, N)
    else:
        n = np.random.randint(args.n_min, args.n_max+1)
        pts = np.random.rand(n, 2)
        G = Grid.unstructured_pts_2d_poisson_dirichlet(pts)
    G.save(os.path.join(args.out_folder, f'{i:04}'))
