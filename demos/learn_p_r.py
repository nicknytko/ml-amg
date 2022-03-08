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
import ns.ga.torch

def parse_bool_str(v):
    v = v.lower()
    if v == 't' or v == 'true':
        return True
    else:
        return False

N = 5
neumann_solve = False

figure_size = (10,10)
alpha = 1. / 3.
n = N**2
grid = ns.model.data.Grid.structured_2d_poisson_dirichlet(N, N)
A = grid.A

np.random.seed(0)
Agg, Agg_roots = pyamg.aggregation.lloyd_aggregation(A, ratio=alpha)

np.random.seed()

device = 'cpu'

A_T = ns.lib.sparse.to_torch_sparse(A).to(device)
A_T_dense = A_T.to_dense()

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


class FeatureMap(nn.Module):
    def __init__(self, in_dim, out_dim, fmap_layers):
        super(FeatureMap, self).__init__()
        hidden = max(in_dim, out_dim)

        self.norm = tgnn.norm.InstanceNorm(in_dim)
        self.nc = tgnn.TAGConv(in_dim, hidden)

        lins = []
        for i in range(fmap_layers-1):
            lins.append(nn.Linear(hidden, hidden))
            lins.append(nn.ReLU())
        lins.append(nn.Linear(hidden, out_dim))
        self.lins = nn.Sequential(*lins)

    def forward(self, x, edge_index, edge_attr):
        if len(x.shape) == 1:
            x = torch.unsqueeze(x, 1)

        x = self.norm(x)
        x = self.nc(x, edge_index, edge_attr)
        x = nnF.relu(x)
        x = self.lins(x)

        return x


class FeatureMapLargeEdge(nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, fmap_layers):
        super(FeatureMapLargeEdge, self).__init__()
        hidden = max(in_dim, out_dim)

        self.norm = tgnn.norm.InstanceNorm(in_dim)
        self.nc = tgnn.NNConv(in_dim, hidden, nn.Sequential(
            nn.Linear(edge_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden*in_dim), nn.ReLU()
        ))

        lins = []
        for i in range(fmap_layers-1):
            lins.append(nn.Linear(hidden, hidden))
            lins.append(nn.ReLU())
        lins.append(nn.Linear(hidden, out_dim))
        self.lins = nn.Sequential(*lins)

    def forward(self, x, edge_index, edge_attr):
        if len(x.shape) == 1:
            x = torch.unsqueeze(x, 1)

        x = self.norm(x)
        x = self.nc(x, edge_index, edge_attr)
        x = nnF.relu(x)
        x = self.lins(x)

        return x


def topk(x, k):
    return torch.argsort(x.squeeze(), descending=True)[:k]


def node_to_edge_ind(edge_index, idx):
    r = []
    node_indices = set(idx.detach().cpu().numpy())
    for e in range(edge_index.shape[1]):
        if edge_index[0, e].item() in node_indices or edge_index[1, e].item() in node_indices:
            r.append(e)
    return torch.Tensor(r).long()


class SolverML(nn.Module):
    def __init__(self, dim, alpha):
        super(SolverML, self).__init__()
        self.full_agg = ns.model.agg_interp.FullAggNet(dim)
        self.alpha = alpha

        self.nc1 = FeatureMapLargeEdge(1, dim, 2, 3)
        self.nc2 = FeatureMapLargeEdge(dim, dim, 2, 3)

        self.nc3 = FeatureMapLargeEdge(dim, dim, 2, 3)
        self.nc4 = FeatureMapLargeEdge(dim, dim, 2, 3)

        self.nc5 = FeatureMapLargeEdge(dim, dim, 2, 3)
        self.nc6 = FeatureMapLargeEdge(dim, 1, 2, 3)

    def coarsen(self, A):
        agg_T, P_T, weights, topk, node_scores = self.full_agg(A, alpha)
        return ns.lib.sparse.torch_to_scipy(P_T)

    def solve(self, A, P, r):
        A_C = P.T@A@P
        if A_C.sum() < 1e-9:
            return torch.Tensor(np.squeeze(np.zeros(r.shape)))

        A_graph = ns.model.data.graph_from_matrix_node_vals_with_inv(A, r)
        A_C_graph = ns.model.data.graph_from_matrix_node_vals_with_inv(A_C.tocsr(), torch.Tensor(P.T@r))
        x, ei, ea = A_graph.x, A_graph.edge_index, A_graph.edge_attr
        ei_C, ea_C = A_C_graph.edge_index, A_C_graph.edge_attr

        # pre conv
        x = nnF.relu(self.nc1(x, ei, ea))
        x = nnF.relu(self.nc2(x, ei, ea))

        # restrict
        x = x.cpu().numpy()
        x = torch.Tensor(P.T@x)

        # coarse conv
        x = nnF.relu(self.nc3(x, ei_C, ea_C))
        x = nnF.relu(self.nc4(x, ei_C, ea_C))

        # interpolate
        x = x.cpu().numpy()
        x = torch.Tensor(P@x)

        # post conv
        x = nnF.relu(self.nc5(x, ei, ea))
        x = nnF.relu(self.nc6(x, ei, ea))

        # print(x.shape)

        return torch.Tensor(np.squeeze(x))


def fitness(gen, weight, idx):
    model.load_state_dict(ns.ga.torch.model_weights_as_dict(model, weight))
    model.eval()

    rand = np.random.RandomState(0)

    iterations = 1
    loss = 0
    for i in range(iterations):
        # solution is random unit vector -> abs error = rel error
        #x_rand = torch.normal(0, 1, (n, 1))
        x_rand = torch.unsqueeze(torch.Tensor(rand.randn(n)), 1)
        x_rand /= torch.norm(x_rand, 2)
        x_rand = x_rand.float()

        #x_init = torch.normal(0, 1, (n, 1)).float()
        x_init = torch.unsqueeze(torch.Tensor(rand.randn(n)),1).float()

        b = A_T@x_rand
        r = b - A_T@x_init
        with torch.no_grad():
            P = model.coarsen(A)
            e = model.solve(A, P, r)
            x_f = x_init + e

        # print(x_f, x_rand)

        loss += torch.norm(x_rand - x_f) / iterations
    return 1./loss


data = ns.model.data.graph_from_matrix_basic(A)
dim = 32
model = SolverML(dim, alpha)

if __name__ == '__main__':
    num_workers = 3
    perturb = 1e-1
    population = ns.ga.torch.TorchGA(model, 20)
    ga = ns.ga.parga.ParallelGA(num_workers=num_workers,
                                initial_population=population.population_weights,
                                model_folds=population.folds,
                                fitness_func=fitness,
                                mutation_probability=0.5,
                                mutation_min_perturb=-perturb,
                                mutation_max_perturb=perturb,
                                steady_state_top_use=3./4.,
                                steady_state_bottom_discard=1./4)
    ga.start_workers()
    print(ga.num_generation, 1./ga.best_solution()[1])
    while True:
        ga.iteration()
        print(ga.num_generation, 1./ga.best_solution()[1])
    ga.finish_workers()
