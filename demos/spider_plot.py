import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import pyamg
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import sys

sys.path.append('../')
import ns.model.data


grid = ns.model.data.Grid.structured_2d_poisson_dirichlet(8, 8)
A = grid.A

n = A.shape[0]
Dinv = sp.diags([1.0 / A.diagonal()], [0])
omega = (4. / 3.) / np.abs(spla.eigs(Dinv @ A, k=1, return_eigenvectors=False)).item()
smoother = (sp.eye(n) - omega*Dinv@A)

AggOp, _ = pyamg.aggregation.lloyd_aggregation(A, ratio=0.2)
P = smoother@AggOp

plt.figure(figsize=(8,8))
ax = plt.gca()
grid.plot(ax)
grid.plot_agg(AggOp, ax, alpha=0.1, edgecolor='0.2')
grid.plot_spider_agg(AggOp, P, ax)
ax.set_aspect('equal')
plt.show()
