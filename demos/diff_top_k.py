import torch
import torch.linalg as tla
import torch.nn as nn
import torch.nn.functional as nnF
import torch_geometric as tg
import torch_geometric.nn as tgnn
import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import sys
import os
import pyamg
import matplotlib.pyplot as plt
import pygad
import pygad.torchga
import argparse
import networkx as nx

sys.path.append('../')
import ns.model.agg_interp
import ns.model.loss
import ns.model.data
import ns.lib.sparse
import ns.lib.sparse_tensor
import ns.lib.multigrid
import ns.lib.graph
import ns.ga.parga

def parse_bool_str(v):
    v = v.lower()
    if v == 't' or v == 'true':
        return True
    else:
        return False

N = 8
neumann_solve = False

figure_size = (10,10)
alpha = 1. / 9.
n = N**2
grid = ns.model.data.Grid.structured_2d_poisson_dirichlet(N, N)
A = grid.A

np.random.seed(0)
Agg, Agg_roots = pyamg.aggregation.lloyd_aggregation(A, ratio=alpha)

np.random.seed()

device = 'cpu'

A_T = ns.lib.sparse.to_torch_sparse(A).to(device)

# Set up Jacobi smoother
Dinv = sp.diags([1.0 / A.diagonal()], [0])
omega = (4. / 3.) / np.abs(spla.eigs(Dinv @ A, k=1, return_eigenvectors=False)).item()
smoother = (sp.eye(n) - omega*Dinv@A)
P_SA = smoother @ Agg

# Create PyTorch tensor versions of everything
smoother_T = ns.lib.sparse.to_torch_sparse(smoother).to(device)
Agg_T = ns.lib.sparse.to_torch_sparse(Agg).to(device)
Agg_roots_T = torch.Tensor(Agg_roots)
A_Graph = ns.model.data.graph_from_matrix_basic(A)

def plot_grid(agg, P, bf_weights, cluster_centers, node_scores):
    graph = grid.networkx
    graph.remove_edges_from(list(nx.selfloop_edges(graph)))
    positions = {}
    for node in graph.nodes:
        positions[node] = grid.x[node]

    edge_values = np.zeros(len(graph.edges))
    for i, edge in enumerate(graph.edges):
        edge_values[i] = bf_weights[edge]

    if isinstance(node_scores, torch.Tensor):
        node_scores = node_scores.numpy()
    #node_scores = np.log10(node_scores + 1)

    if not isinstance(P, sp.spmatrix):
        P = ns.lib.sparse_tensor.to_scipy(P)

    plt.gcf().set_size_inches(figure_size[0], figure_size[1])
    nx.drawing.nx_pylab.draw_networkx(graph, ax=plt.gca(), pos=positions, arrows=False, with_labels=False, node_size=60, edge_color=edge_values, node_color=node_scores)
    grid.plot_agg(agg, alpha=0.1, edgecolor='0.2')
    #grid.plot_spider_agg(agg, P)
    plt.plot(grid.x[cluster_centers, 0], grid.x[cluster_centers, 1], 'y*', markersize=10)
    plt.gca().set_aspect('equal')

def display_progress(ga_instance):
    weights = ga_instance.best_solution()
    gen = ga_instance.num_generation

    agg, P, bf_weights, cluster_centers, node_scores = compute_agg_and_p(weights[0])
    agg = ns.lib.sparse_tensor.to_scipy(agg)

    print(f'Generation = {gen}')
    print(f'Fitness    = {weights[1]}')
    print(f'Loss       = {1.0/weights[1]}')

    plt.clf()
    plot_grid(agg, P, bf_weights, cluster_centers, node_scores)
    plt.title(f'Generation {gen}, {title}, conv={1.0/weights[1]:.4f}')
    plt.pause(0.1)
    plt.savefig(f'{fig_directory}/{gen}_agg.pdf')

plt.ion()
plt.figure(figsize=figure_size)
plot_grid(Agg, P_SA, A, Agg_roots, torch.zeros(A.shape[0]))
plt.title('Baseline')
plt.pause(0.5)

def diff_top_k(x, k, alpha=0.1):
    top_k = torch.argsort(x, descending=True)[:k]
    top_k_vec = torch.ones(x.shape) * -1
    top_k_vec[top_k] = 1.0

    k_smallest = abs(x[top_k[k-1]])

    return top_k_vec, torch.sigmoid((x-k_smallest) + top_k_vec*alpha)

agg_vec = torch.zeros(Agg_T.shape[0])
agg_vec[Agg_roots_T.long()] = 1.0
# print(agg_vec)

x = torch.rand(Agg_T.shape[0], requires_grad=True)
k = int(x.shape[0] * alpha)
# print(x, k)
# print(diff_top_k(x, k))

class SoftAggLayer(nn.Module):
    def __init__(self, first_layer=False, node_activation=nn.ReLU()):
        super(SoftAggLayer, self).__init__()

        self.nc1 = tg.nn.TAGConv(1 if first_layer else 2, 8, K=3)
        self.nc2 = tg.nn.TAGConv(8, 8, K=3)
        self.nc3 = tg.nn.TAGConv(8, 1, K=3)
        self.norm = tg.nn.InstanceNorm(1)

    def forward(self, x, edge_index, edge_attr, k):
        if len(x.shape) == 1:
            x = torch.unsqueeze(x, 1)

        x = nnF.relu(self.nc1(x, edge_index, abs(edge_attr)))
        x = nnF.relu(self.nc2(x, edge_index, abs(edge_attr)))
        x = nnF.relu(self.nc3(x, edge_index, abs(edge_attr)))
        x = self.norm(x).squeeze()
        return torch.column_stack((x, diff_top_k(x, k)[1]))

class SoftAggNet(nn.Module):
    def __init__(self, iterations=4):
        super(SoftAggNet, self).__init__()
        layers = []
        for i in range(iterations):
            layers.append(SoftAggLayer(i == 0))
        self.layers = nn.ModuleList(layers)

    def forward(self, D, k):
        x, edge_index, edge_attr = D.x, D.edge_index, D.edge_attr
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, k)
        return x[:,1], edge_attr

data = ns.model.data.graph_from_matrix_basic(A)
model = SoftAggNet(iterations=10)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

i=0
while True:
    #y = diff_top_k(x, k)[1]
    y, edge = model(data, k)
    loss = torch.sum((agg_vec - y) ** 2)
    loss.backward()
    optimizer.step()
    if i % 10 == 0:
        print(y)
        plt.clf()
        plt.title(f'Iteration {i}: {loss}')
        plot_grid(Agg, P_SA, A, Agg_roots, y.detach().numpy())
        plt.pause(0.001)
    i+=1

# print(x)
# print(y)
# print(agg_vec)
