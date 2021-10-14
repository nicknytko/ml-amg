import torch as T
import torch
import torch_geometric
import torch.optim as optim
import copy
import networkx as nx
from torch.nn import ReLU, GRU, Sequential, Linear
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
import torch.nn as nn
from torch_geometric.nn import (NNConv, GATConv, graclus, max_pool, max_pool_x,
                                global_mean_pool, BatchNorm, InstanceNorm, GraphConv,
                                GCNConv, TAGConv, SGConv, LEConv, TransformerConv, SplineConv,
                                GMMConv, GatedGraphConv, ARMAConv, GENConv, DeepGCNLayer,
                                LayerNorm, GraphUNet)
import torch_geometric.utils.convert as tgc
from torch_geometric.data import Data
import scipy.sparse as sp


def activation_func(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['sigmoid', nn.Sigmoid()],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]


class smallEdgeModel(nn.Module):

    def __init__(self, in_dim, hid_dim, out_dim):
        super(smallEdgeModel, self).__init__()
        self.edge_mlp = Seq(Lin(in_dim, hid_dim),   ReLU(), torch.nn.LayerNorm([hid_dim]),
                            Lin(hid_dim, out_dim))


    def forward(self, src, dest, edge_attr):#, u, batch):
        # src, dest: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.

        out = torch.cat([src, dest, edge_attr],1)#, u[batch]], 1)
        out = self.edge_mlp(out)
        return out


class EdgeModel(nn.Module):

    def __init__(self, in_dim, dims, out_dim):
        super(EdgeModel, self).__init__()

        blocks = []
        activations = []
        normalizations = []
        param_block = []

        this_block = [Lin(in_dim, dims[0])]
        param_block.extend(this_block)
        blocks.append(Seq(*this_block))
        activations.append(activation_func('relu'))
        normalizations.append(nn.LayerNorm(dims[0]))

        for i in range(len(dims)-1):
            this_block = [Lin(dims[i], dims[i+1]), ReLU(), Lin(dims[i+1], dims[i+1])]

            param_block.extend(this_block)
            blocks.append(Seq(*this_block))
            activations.append(activation_func('relu'))
            normalizations.append(nn.LayerNorm(dims[i+1]))

        this_block = [Lin(dims[-1], out_dim)]
        param_block.extend(this_block)
        blocks.append(Seq(*this_block))

        activations.append(activation_func('none'))
        normalizations.append(activation_func('none'))

        self.blocks = nn.ModuleList(blocks)
        self.activate = nn.ModuleList(activations)
        self.normaliz = nn.ModuleList(normalizations)
        self.network = Seq(*param_block)

    def forward(self, src, dest, edge_attr):#, u, batch):
        # src, dest: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.

        x = torch.cat([src, dest, edge_attr],1)#, u[batch]], 1)

        for block, activate, normalization in zip(self.blocks, self.activate, self.normaliz):
            residual = x
            x = block(x)

            if normalization is not None:
                x = normalization(x)
            if x.shape == residual.shape:
                x += residual

            x = activate(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, res_layers, dim, K, lr):
        super().__init__()
        self.in_channels, self.out_channels, self.res_layers, self.dim, self.K, self.lr = \
            in_channels, out_channels, res_layers, dim, K, lr

        blocks = []
        activations = []
        normalizations = []
        param_block = []

        this_block = [(TAGConv(in_channels, dim[0], K = K), 'x, edge_index, edge_attr -> x')]
        param_block.append(*this_block)
        blocks.append(torch_geometric.nn.Sequential('x, edge_index, edge_attr', this_block))

        activations.append(activation_func('relu'))
        normalizations.append(torch_geometric.nn.norm.InstanceNorm(dim[0]))

        for i in range(res_layers):

            this_block = [(TAGConv(dim[i], dim[i+1], K = K), 'x, edge_index, edge_attr -> x'), ReLU(),
                                 (TAGConv(dim[i+1], dim[i+1], K = K), 'x, edge_index, edge_attr -> x'),]
            for sub_block in this_block:
                param_block.append(sub_block)

            blocks.append(torch_geometric.nn.Sequential('x, edge_index, edge_attr', this_block))
            activations.append(activation_func('relu'))
            normalizations.append(torch_geometric.nn.norm.InstanceNorm(dim[i+1]))

        this_block = [(TAGConv(dim[-1], dim[-1], K=K), 'x, edge_index, edge_attr -> x')]
        param_block.append(*this_block)
        blocks.append(torch_geometric.nn.Sequential('x, edge_index, edge_attr', this_block))

        activations.append(activation_func('relu'))
        normalizations.append(activation_func('none'))

        self.edge_model = EdgeModel(dim[-1]*2+1, dim[::-1], 1)
        self.blocks = blocks
        self.activate = activations
        self.normaliz = normalizations
        self.network = torch_geometric.nn.Sequential('x, edge_index, edge_attr', param_block)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, D):
        data, edge_index, edge_attr = D.x, D.edge_index, D.edge_attr
        edge_attr = edge_attr.flatten()  # uncomment for TAG conv

        for block, activate, normalization in zip(self.blocks, self.activate, self.normaliz):
            residual = data
            if normalization is not None:
                data = normalization(block(data, edge_index, edge_attr))
            else:
                data = block(data, edge_index, edge_attr)

            if data.shape == residual.shape:
                data += residual

            data = activate(data)

        row = edge_index[0]
        col = edge_index[1]

        edge_attr = edge_attr.unsqueeze(1).float()
        edge_attr = self.edge_model(data[row], data[col], edge_attr)#, u,
                                        # batch if batch is None else batch[row])

        edge_attr = (edge_attr - edge_attr.flatten().mean())/edge_attr.std()
        return abs(edge_attr)


class GraphNet(nn.Module):
    def __init__(self,  K, lr):
        super().__init__()
        self.K, self.lr =  K, lr

        this_block1 = [(TAGConv(1, 8, K = K), 'x, edge_index, edge_attr -> x'), ReLU(),
                                 (TAGConv(8, 32, K = K), 'x, edge_index, edge_attr -> x')]

        self.node_block1 = torch_geometric.nn.Sequential('x, edge_index, edge_attr', this_block1)
        self.edge_net1 = smallEdgeModel(32*2+1, 8, 1)

        this_block2 = [(TAGConv(32, 64, K = K), 'x, edge_index, edge_attr -> x'), ReLU(),
                                 (TAGConv(64, 128, K = K), 'x, edge_index, edge_attr -> x')]

        self.node_block2 = torch_geometric.nn.Sequential('x, edge_index, edge_attr', this_block2)
        self.edge_net2 = smallEdgeModel(128*2+1, 32, 1)

        this_block3 = [(TAGConv(128, 256, K = K), 'x, edge_index, edge_attr -> x'), ReLU(),
                                 (TAGConv(256, 512, K = K), 'x, edge_index, edge_attr -> x')]

        self.node_block3 = torch_geometric.nn.Sequential('x, edge_index, edge_attr', this_block3)
        self.edge_net3 = smallEdgeModel(512*2+1, 128, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, D):
        x, edge_index, edge_attr = D.x, D.edge_index, D.edge_attr
        x = self.node_block1(x, edge_index, edge_attr)

        row = edge_index[0]
        col = edge_index[1]

        edge_attr = edge_attr.unsqueeze(1).float()
        edge_attr = self.edge_net1(x[row], x[col], edge_attr).squeeze()

        x = self.node_block2(x, edge_index, edge_attr)

        edge_attr = edge_attr.unsqueeze(1).float()
        edge_attr = self.edge_net2(x[row], x[col], edge_attr)
        edge_attr = edge_attr.squeeze()  # uncomment for TAG conv

        x = self.node_block3(x, edge_index, edge_attr)

        edge_attr = edge_attr.unsqueeze(1).float()
        edge_attr = self.edge_net3(x[row], x[col], edge_attr)
        edge_attr = (edge_attr - edge_attr.squeeze().mean())/edge_attr.std()

        return abs(edge_attr)


class InterpolationNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.hops = 5
        self.lr = 0.001
        self.model = ResidualBlock(1, 1, 9, [16, 16, 32, 32, 64, 64, 128, 128, 256, 256], self.hops, self.lr)

    def graph_from_matrix(self, A, C, F):
        n = A.shape[1]

        g = nx.from_scipy_sparse_matrix(A)
        fine_nodes = list(set(range(n)) - set(C))
        Hc = g.subgraph(C)
        coarse2_remove_edge = Hc.edges
        g.remove_edges_from(coarse2_remove_edge)

        Hf = g.subgraph(F)
        fine2_remove_edge = Hf.edges
        g.remove_edges_from(fine2_remove_edge)
        data = tgc.from_networkx(g)

        edge_attr  = abs(data.weight)
        edge_index = data.edge_index
        x = torch.zeros(n)
        x[C] = 1.0
        x = x.unsqueeze(1).float()
        output = Data(x=x, edge_index=edge_index, edge_attr= edge_attr.float())

        return output

    def adapt_dim(self, shape, indices, P):
        values = P.flatten().cpu().numpy()
        indices = indices.cpu().numpy()

        coo = sp.coo_matrix((values, (indices[0], indices[1])), shape=shape)
        return coo.tolil()

    def forward(self, G, A, C, F):
        n = A.shape[1]
        n_g = G.x.shape[0]

        # Output prolongation weights
        with torch.no_grad():
            P = self.model(G)
        # Form nxn matrix
        P = self.adapt_dim((n_g, n_g), G.edge_index, P)
        # Update diagonal
        for j in C:
            P[j,j] = 1.
        # Remove fine columns
        P = P.tocsc()
        P = P[:, C]
        # Return final
        return P.tocsr()

    def forward_mat(self, A, C, F):
        G = self.graph_from_matrix(A, C, F)
        return self.forward(G, A, C, F)
