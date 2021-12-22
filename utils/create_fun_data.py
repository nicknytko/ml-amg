import numpy as np
import numpy.linalg as la
import scipy.stats.qmc
import argparse
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
import ns.model.data

parser = argparse.ArgumentParser(description='Create fun example grids')
parser.add_argument('system', choices=['circle'])
parser.add_argument('n', type=int, help='Number of points')
parser.add_argument('--out', type=str, default='out.grid', help='Output file name')
args = parser.parse_args()

num_pts = args.n

def sunflower(n, alpha):
    '''
    https://stackoverflow.com/questions/28567166/uniformly-distribute-x-points-inside-a-circle
    '''
    b = int(alpha * (n ** 0.5))
    phi = (5**0.5 + 1)/2

    points = np.zeros((n, 2))
    for i in range(n):
        if i > n - b:
            r = 1
        else:
            r = ((i - 0.5) ** 0.5) / ((n - ((b+1) / 2)) ** 0.5)
        r = np.real(r)
        theta = (2 * np.pi * i) / phi**2
        points[i] = np.array([r * np.cos(theta), r * np.sin(theta)])
    return points

points = sunflower(num_pts, 2)

plt.figure(figsize=(10,10))
grid = ns.model.data.Grid.unstructured_pts_2d_poisson_dirichlet(points)
grid.plot()
grid.save(args.out)
plt.axis('equal')
plt.show()
