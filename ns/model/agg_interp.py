import torch
import torch_geometric as tg
import torch.nn as nn
import torch.nn.functional as nnF
import ns.lib.sparse as sparse

class TensorLambda(nn.Module):
    '''
    Small helper to perform some function in a Sequential block
    '''
    def __init__(self, func):
        super(TensorLambda, self).__init__()
        self.f = func

    def forward(self, x):
        return self.f(x)

class smallEdgeModel(nn.Module):
    '''
    Small edge convolution model, borrowed/stolen from Ali's code.
    Thanks, Ali!
    '''
    def __init__(self, in_dim, hid_dim, out_dim):
        super(smallEdgeModel, self).__init__()
        self.edge_mlp = nn.Sequential(nn.Linear(in_dim, hid_dim), nn.ReLU(),
                                      nn.LayerNorm([hid_dim]),
                                      nn.Linear(hid_dim, out_dim))

    def forward(self, src, dest, edge_attr):#, u, batch):
        # src, dest: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.

        out = torch.cat([src, dest, edge_attr],1)#, u[batch]], 1)
        out = self.edge_mlp(out)
        return out

class MPNN(nn.Module):
    def __init__(self, dim):
        super(MPNN, self).__init__()

        self.conv1 = tg.nn.NNConv(1, dim, nn=nn.Sequential(
            TensorLambda(lambda x: x.reshape(-1, 2)),
            nn.Linear(2, 4), nn.ReLU(),
            nn.Linear(4, 16), nn.ReLU(),
            nn.Linear(16, 1 * dim), nn.ReLU()
        ))
        self.normalize1 = tg.nn.norm.InstanceNorm(dim)

        self.conv2 = tg.nn.NNConv(dim, dim, nn=nn.Sequential(
            TensorLambda(lambda x: x.reshape(-1, 2)),
            nn.Linear(2, 4), nn.ReLU(),
            nn.Linear(4, 16), nn.ReLU(),
            nn.Linear(16, dim * dim), nn.ReLU()
        ))
        self.normalize2 = tg.nn.norm.InstanceNorm(dim)

        self.conv3 = tg.nn.NNConv(dim, dim, nn=nn.Sequential(
            TensorLambda(lambda x: x.reshape(-1, 2)),
            nn.Linear(2, 4), nn.ReLU(),
            nn.Linear(4, 16), nn.ReLU(),
            nn.Linear(16, dim * dim), nn.ReLU()
        ))
        self.normalize3 = tg.nn.norm.InstanceNorm(dim)

        self.conv4 = tg.nn.NNConv(dim, dim, nn=nn.Sequential(
            TensorLambda(lambda x: x.reshape(-1, 2)),
            nn.Linear(2, 4), nn.ReLU(),
            nn.Linear(4, 16), nn.ReLU(),
            nn.Linear(16, dim * dim), nn.ReLU()
        ))
        self.normalize4 = tg.nn.norm.InstanceNorm(dim)

        self.conv5 = tg.nn.NNConv(dim, dim, nn=nn.Sequential(
            TensorLambda(lambda x: x.reshape(-1, 2)),
            nn.Linear(2, 4), nn.ReLU(),
            nn.Linear(4, 16), nn.ReLU(),
            nn.Linear(16, dim * dim), nn.ReLU()
        ))
        self.normalize5 = tg.nn.norm.InstanceNorm(dim)

        self.conv6 = tg.nn.NNConv(dim, 1, nn=nn.Sequential(
            TensorLambda(lambda x: x.reshape(-1, 2)),
            nn.Linear(2, 4), nn.ReLU(),
            nn.Linear(4, 16), nn.ReLU(),
            nn.Linear(16, dim * 1), nn.ReLU()
        ))
        self.normalize6 = tg.nn.norm.InstanceNorm(dim)

        self.edge_model1 = smallEdgeModel(dim*2+2, dim, 2)
        self.edge_model2 = smallEdgeModel(dim*2+2, dim, 2)
        self.edge_model3 = smallEdgeModel(dim*2+2, dim, 2)
        self.edge_model4 = smallEdgeModel(dim*2+2, dim, 2)
        self.edge_model5 = smallEdgeModel(dim*2+2, dim, 2)
        self.edge_model6 = smallEdgeModel(dim*2+2, dim, 1)


    def forward(self, D):
        x, edge_index, edge_attr = D.x, D.edge_index, D.edge_attr
        x = x.reshape((-1, 1))

        row = edge_index[0]
        col = edge_index[1]

        # conv 1
        x = nnF.relu(self.conv1(self.normalize1(x), edge_index, edge_attr)) + x
        edge_attr = nnF.relu(self.edge_model1(x[row], x[col], edge_attr.float())) + edge_attr

        # conv 2
        x = nnF.relu(self.conv2(self.normalize2(x), edge_index, edge_attr)) + x
        edge_attr = nnF.relu(self.edge_model2(x[row], x[col], edge_attr.float())) + edge_attr

        # conv 3
        x = nnF.relu(self.conv3(self.normalize3(x), edge_index, edge_attr)) + x
        edge_attr = nnF.relu(self.edge_model3(x[row], x[col], edge_attr.float())) + edge_attr

        # conv 4
        x = nnF.relu(self.conv4(self.normalize4(x), edge_index, edge_attr)) + x
        edge_attr = nnF.relu(self.edge_model4(x[row], x[col], edge_attr.float())) + edge_attr

        # conv 5
        x = nnF.relu(self.conv5(self.normalize5(x), edge_index, edge_attr)) + x
        edge_attr = nnF.relu(self.edge_model5(x[row], x[col], edge_attr.float())) + edge_attr

        # conv 6
        x = nnF.relu(self.conv6(self.normalize6(x), edge_index, edge_attr)) + x
        edge_attr = nnF.relu(self.edge_model6(x[row], x[col], edge_attr.float()))

        return edge_attr


class ColumnWeightsNetwork(nn.Module):
    def __init__(self):
        super(ColumnWeightsNetwork, self).__init__()

        self.conv1 = tg.nn.NNConv(1, 4, nn=nn.Sequential(
            TensorLambda(lambda x: x.reshape(-1, 3)),
            nn.Linear(3, 4), nn.ReLU()
        ))
        self.act1 = nn.ReLU()
        self.conv2 = tg.nn.NNConv(4, 1, nn=nn.Sequential(
            TensorLambda(lambda x: x.reshape(-1, 3)),
            nn.Linear(3, 4), nn.ReLU()
        ))
        self.act2 = nn.Sigmoid()


    def forward(self, x, edge_index, edge_attr):
        x = x.reshape((-1, 1))
        x1 = self.act1(self.conv1(x, edge_index, edge_attr))
        x2 = self.act2(self.conv2(x1, edge_index, edge_attr))
        return x2

class PNet(nn.Module):
    def __init__(self, dim, device):
        super(PNet, self).__init__()

        self.P_values = MPNN(dim)
        self.P_column_weights = ColumnWeightsNetwork()
        self.device = device
        self.to(device)

    def forward_P_hat(self, data):
        P_hat_vals = self.P_values.forward(data).squeeze()
        P_hat = torch.sparse_coo_tensor(data.edge_index, P_hat_vals, (data.x.shape[0], data.x.shape[0])).to(self.device)

        return P_hat.coalesce()

    def forward(self, data, agg_csr):
        agg_T = sparse.to_torch_sparse(agg_csr).to(self.device)

        # First, compute \hat{P}, or some matrix w/ same sparsity as A
        P_hat_vals = self.P_values.forward(data).squeeze()
        P_hat = torch.sparse_coo_tensor(data.edge_index, P_hat_vals, (data.x.shape[0], data.x.shape[0])).to(self.device)

        # Find the optimal column re-weightings for Agg, and compute \hat{Agg}
        # agg_weight_values = self.P_column_weights(agg_T.values(), network_data.edge_index, network_data.edge_attr).squeeze()
        # agg_weighted = torch.sparse_coo_tensor(agg_T.indices(), agg_weight_values, agg_T.size()).to(self.device)

        # Compute P := \hat{P} \hat{Agg}
        return torch.sparse.mm(P_hat, agg_T).coalesce()
