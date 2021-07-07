import numpy as np
import pickle
import torch
import torch._six
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


def scipy_csr_to_pytorch_sparse(A):
    Acoo = A.tocoo()
    indices = np.row_stack([Acoo.row, Acoo.col])
    values = Acoo.data
    shape = Acoo.shape
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)


class ShapeDebugger(nn.Module):
    def __init__(self, msg=''):
        super(ShapeDebugger, self).__init__()
        self.msg = msg

    def forward(self, x):
        print(f' {self.msg}: {x.shape}')
        return x


class TensorLambda(nn.Module):
    def __init__(self, func):
        super(TensorLambda, self).__init__()
        self.f = func

    def forward(self, x):
        return self.f(x)


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


class GNN(nn.Module):
    _default_state_fname = 'out_model/mpnn'

    def __init__(self, device):
        super(GNN, self).__init__()
        self.device = device

        self.conv1 = tg.nn.TAGConv(1, 20, K=30)
        self.conv2 = tg.nn.TAGConv(20, 20, K=30)
        self.conv3 = tg.nn.TAGConv(20, 20, K=30)
        self.conv4 = tg.nn.TAGConv(20, 19, K=30)

        self.lin1 = nn.Linear(80, 60)
        self.lin2 = nn.Linear(60, 40)
        self.lin3 = nn.Linear(40, 30)
        self.lin4 = nn.Linear(30, 10)
        self.lin5 = nn.Linear(10, 1)


    def load_from_file(self, fname=None):
        if fname is None:
            fname = GNN._default_state_fname
        self.load_state_dict(torch.load('model/mpnn'))
        self.eval()


    def save_to_file(self, fname=None):
        if fname is None:
            fname = GNN._default_state_fname
        torch.save(self.state_dict, fname)


    def forward(self, data):
        edge_index = data.edge_index
        edge_weight = data.edge_weight
        edge_index = edge_index.to(self.device)
        edge_weight = edge_weight.float().to(self.device)
        x = data.x

        x = x.reshape((-1, 1)).float()

        x2 = nnF.relu(self.conv1(x, edge_index, edge_weight)).float()
        x3 = nnF.relu(self.conv2(x2, edge_index, edge_weight)).float()
        x4 = nnF.relu(self.conv3(x3, edge_index, edge_weight)).float()
        x5 = nnF.relu(self.conv4(x4, edge_index, edge_weight)).float()

        res_stack = torch.cat((x, x2, x3, x4, x5), 1)
        res_nn = nnF.relu(self.lin1(res_stack))
        res_nn = nnF.relu(self.lin2(res_nn))
        res_nn = nnF.relu(self.lin3(res_nn))
        res_nn = nnF.relu(self.lin4(res_nn))
        res_nn = nnF.relu(self.lin5(res_nn))

        return nnF.relu(tg.nn.global_mean_pool(res_nn.reshape(-1), data.batch))


class MeshDataset(tg.data.Dataset):
    def __init__(self):
        super(MeshDataset, self).__init__()

        self.grids = np.array(helpers.pickle_load_bz2(f'../out_grids/output-splittings.pkl.bz2'))
        self.convs = np.array(helpers.pickle_load_bz2(f'../out_grids/output-conv.pkl.bz2'))
        self.matnames = np.array(helpers.pickle_load_bz2(f'../out_grids/output-matnames.pkl.bz2'))

        unique_matnames = np.unique(self.matnames)
        self.As = {}
        for mat in unique_matnames:
            loaded_mat = sio.loadmat(f'../out_matrices/{mat}')
            self.As[mat] = loaded_mat['A']

    def __len__(self):
        return len(self.grids)

    def get(self, idx):
        grid = self.grids[idx]
        conv = self.convs[idx]
        A = self.As[self.matnames[idx]]

        edge_index, edge_weight = tg.utils.from_scipy_sparse_matrix(helpers.normalize_mat(A))
        in_val = np.array(grid).astype(np.float32)
        x = torch.from_numpy(np.array(grid))
        return tg.data.Data(x=x,
                            edge_index=edge_index,
                            edge_weight=edge_weight,
                            y=torch.from_numpy(np.array([conv]).astype(np.float32)).reshape(1))
