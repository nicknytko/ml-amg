import torch
import torch.linalg as tla
import torch.nn as nn
import torch_geometric.nn as tgnn
import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import sys
import sys
import matplotlib.pyplot as plt
import matplotlib.colors
import argparse
import networkx as nx

sys.path.append('../')
import ns.model.data

def parse_bool_str(v):
    v = v.lower()
    if v == 't' or v == 'true':
        return True
    else:
        return False

parser = argparse.ArgumentParser(description='Demo of the aggregate-picking network')
parser.add_argument('system', type=str, help='Grid file to visualize')
parser.add_argument('--save-figure', type=str, default=None)
args = parser.parse_args()

grid = ns.model.data.Grid.load(args.system)
graph = grid.networkx
graph.remove_edges_from(list(nx.selfloop_edges(graph)))
positions = {}
for node in graph.nodes:
    positions[node] = grid.x[node]

print(grid.extra)

if 'jumps' in grid.extra:
    J = grid.extra['jumps']
    print(J)
    S = J[:, :2]
    D = J[:, 2]
    Ns = J.shape[0]

    X = grid.x.reshape((1, -1, 2))
    Xr = np.transpose(np.repeat(X, Ns, axis=0), axes=(1,0,2))
    assignment = np.argmin(la.norm(Xr-S, axis=2), axis=1)
    dd = D[assignment]

    nx.drawing.nx_pylab.draw_networkx_edges(graph, ax=plt.gca(), pos=positions, arrows=False)
    plt.scatter(grid.x[:,0], grid.x[:,1], c=dd, norm=matplotlib.colors.LogNorm())
    plt.colorbar()
elif 'noise' in grid.extra:
    noise = grid.extra['noise']
    nx.drawing.nx_pylab.draw_networkx_edges(graph, ax=plt.gca(), pos=positions, arrows=False)
    plt.scatter(grid.x[:,0], grid.x[:,1], c=10.**noise, norm=matplotlib.colors.LogNorm())
    plt.colorbar()
else:
    nx.drawing.nx_pylab.draw_networkx(graph, ax=plt.gca(), pos=positions, arrows=False, with_labels=False, node_size=20)

#grid.plot(node_size=20)
plt.gca().set_aspect('equal')
plt.gca().axis('off')
if args.save_figure is not None:
    plt.savefig(args.save_figure)
plt.show(block=True)


print('DoF', grid.A.shape[0])
