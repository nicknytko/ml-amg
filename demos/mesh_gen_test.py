import pygmsh
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as spat

N = 100
X = np.random.rand(N, 2)

hull = spat.ConvexHull(X)
bv = X[hull.vertices]
with pygmsh.geo.Geometry() as geom:
    geom.add_polygon(bv, mesh_size=0.5)
    mesh = geom.generate_mesh()

V = mesh.points
plt.plot(V[:,0], V[:,1], 'o')
plt.triplot(V[:,0], V[:,1], mesh.cells_dict['triangle'])

boundary_points = set()
for (fr, to) in mesh.cells_dict['line']:
    boundary_points.add(fr)
    boundary_points.add(to)
boundary_points = list(boundary_points)

print('number of points', len(V))

plt.plot(V[boundary_points, 0], V[boundary_points, 1], 'o')
plt.show(block=True)
