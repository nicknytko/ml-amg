import numpy as np
import pickle
import torch
import torch.linalg as tla
import torch.nn as nn
import torch.nn.functional as nnF
import torch.utils.data as td
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch_geometric as tg
import scipy.io as sio
import sys
from rangedict import RangeDict
from ns.lib.multigrid import *
import ns.lib.helpers as helpers


def mat_to_graph(A):
    edge_index, edge_weight = tg.utils.from_scipy_sparse_matrix(helpers.normalize_mat(A))
    return tg.data.Data(edge_index=edge_index,
                        edge_weight=edge_weight,
                        x=torch.ones(A.shape[0]))


class FullyConnected(nn.Module):
    def __init__(self, N, device):
        super(FullyConnected, self).__init__()
        self.N = N

        layer_sizes = np.linspace(N, 1, 8)

        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(int(layer_sizes[i]), int(layer_sizes[i+1])))
            layers.append(nn.ReLU())
        self.linear_layers = nn.Sequential(*layers)


    def forward(self, data):
        x = data.x.reshape((-1, self.N)).float()
        return self.linear_layers(x)


class InterpolationNetwork(nn.Module):
    def __init__(self, device):
        super(InterpolationNetwork, self).__init__()
        self.device = device

        self.conv1 = tg.nn.TAGConv(2, 15, K=50)
        self.conv2 = tg.nn.TAGConv(15, 30, K=50)
        self.conv3 = tg.nn.TAGConv(30, 15, K=50)
        self.conv4 = tg.nn.TAGConv(15, 1, K=50)

    def forward(self, data, c, i):
        edge_index = data.edge_index
        edge_weight = data.edge_weight.float()

        cv = torch.zeros(data.x.shape)
        cv[i] = 1.0
        inpt = torch.column_stack([cv.to(self.device), c.flatten().to(self.device)]).float()

        x2 = nnF.relu(self.conv1(inpt, edge_index, edge_weight)).float()
        x3 = nnF.relu(self.conv2(x2, edge_index, edge_weight)).float()
        x4 = nnF.relu(self.conv3(x3, edge_index, edge_weight)).float()
        x5 = nnF.relu(self.conv4(x4, edge_index, edge_weight)).float()

        return torch.squeeze(x5)

class VecToFloat(nn.Module):
    def __init__(self):
        super(VecToFloat, self).__init__()

    def forward(self, x):
        return x.float()

class CoarseFineNetwork(nn.Module):
    def __init__(self, device):
        super(CoarseFineNetwork, self).__init__()
        self.device = device

        self.conv = tg.nn.Sequential('x, edge_index, edge_weight', [
            (tg.nn.TAGConv(1,   60,  K=20), 'x, edge_index, edge_weight -> x'), nn.ReLU(), VecToFloat(),
            (tg.nn.TAGConv(60,  100, K=20), 'x, edge_index, edge_weight -> x'), nn.ReLU(), VecToFloat(),
            (tg.nn.TAGConv(100, 200, K=20), 'x, edge_index, edge_weight -> x'), nn.ReLU(), VecToFloat(),
            (tg.nn.TAGConv(200, 80,  K=20), 'x, edge_index, edge_weight -> x'), nn.ReLU(), VecToFloat(),
            (tg.nn.TAGConv(80,  1,   K=20), 'x, edge_index, edge_weight -> x'), nn.Sigmoid(), VecToFloat()
        ])

    def forward(self, data):
        edge_index = data.edge_index
        edge_weight = data.edge_weight.float()

        x = data.x.flatten().float().unsqueeze(1)
        return torch.squeeze(self.conv(x, edge_index, edge_weight))


class ContinuousInterpolationFullNetwork(nn.Module):
    def __init__(self, device):
        super(ContinuousInterpolationFullNetwork, self).__init__()
        self.device = device
        self.P = InterpolationNetwork(device).to(device)
        self.CF = CoarseFineNetwork(device).to(device)

    def forward_P(self, data, c):
        C = torch.diag(c)
        n = len(c)

        P = []
        Phat = []
        for i in range(n):
            z = torch.zeros(n)
            z[i] = 1.0
            p_i = self.P(data, c, i)
            Phat.append(p_i)

            if c[i] >= 0.5:
                P.append(p_i)

        if len(P) > 0:
            P = torch.column_stack(P)
        else:
            P = None

        Phat = torch.column_stack(Phat)
        return P, Phat, C

    def forward(self, data):
        c = self.CF(data)
        return self.forward_P(data, c)

def R_jacobi(A, omega=0.666, nu=5):
    D = np.array(A.diagonal())
    Dinv = 1.0 / D
    J_ep = sp.eye(A.shape[0]) - omega * sp.spdiags(Dinv, [0], m=A.shape[0], n=A.shape[1]) @ A

    J_epn = sp.eye(A.shape[0])
    for i in range(nu):
        J_epn = J_epn @ J_ep

    J_T = torch.Tensor(J_epn.todense())
    return J_T

def E_loss_discrete(A, P, R):
    I = torch.eye(A.shape[0]).to(A.device)
    G = I - P @ tla.solve(P.T @ A @ P, P.T @ A)
    E = R @ G @ R
    return torch.norm(E, 'fro') ** 2

def EC_loss(A, Phat, C, R):
    Pbar = Phat @ C
    I = torch.eye(A.shape[0]).to(A.device)
    Gbar = I - Pbar @ tla.solve((Pbar.T @ A @ Pbar) + I - C, Pbar.T @ A)
    E = R @ Gbar @ R
    c = torch.diagonal(C)
    c_nrm = tla.norm(torch.diagonal(C), ord=1)
    return (torch.norm(E, 'fro') ** 2 +
            (tla.norm(c, ord=1) * 0.001) +
            (tla.norm((1-c) * c, ord=2)) * 0.01)
