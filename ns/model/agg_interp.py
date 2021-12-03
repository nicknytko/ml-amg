import torch
import torch_geometric as tg
import torch.nn as nn
import torch.nn.functional as nnF
import numpy as np
import scipy.sparse as sp
import ns.lib.sparse as sparse
import ns.model.data

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
    def __init__(self, dim, node_activation=nn.ReLU(), edge_activation=nn.ReLU(), num_internal_conv=4):
        super(MPNN, self).__init__()

        # input -> hidden dim layer
        self.node_conv_in = tg.nn.NNConv(1, dim, nn=nn.Sequential(
            TensorLambda(lambda x: x.reshape(-1, 2)),
            nn.Linear(2, 4), nn.ReLU(),
            nn.Linear(4, 16), nn.ReLU(),
            nn.Linear(16, 1 * dim), nn.ReLU()
        ))
        self.normalize_in = tg.nn.norm.InstanceNorm(dim)
        self.edge_conv_in = smallEdgeModel(dim*2+2, dim, 2)

        # Create 'n' internal layers
        node_convs = []
        edge_convs = []
        normalizations = []
        for i in range(num_internal_conv):
            node_convs.append(
                tg.nn.NNConv(dim, dim, nn=nn.Sequential(
                    TensorLambda(lambda x: x.reshape(-1, 2)),
                    nn.Linear(2, 4), nn.ReLU(),
                    nn.Linear(4, 16), nn.ReLU(),
                    nn.Linear(16, dim * dim), nn.ReLU()
                ))
            )
            edge_convs.append(smallEdgeModel(dim*2+2, dim, 2))
            normalizations.append(tg.nn.norm.InstanceNorm(dim))
        self.node_convs = nn.ModuleList(node_convs)
        self.edge_convs = nn.ModuleList(edge_convs)
        self.normalizations = nn.ModuleList(normalizations)
        self.num_internal_conv = num_internal_conv

        # dim -> output layer
        self.node_conv_out = tg.nn.NNConv(dim, 1, nn=nn.Sequential(
            TensorLambda(lambda x: x.reshape(-1, 2)),
            nn.Linear(2, 4), nn.ReLU(),
            nn.Linear(4, 16), nn.ReLU(),
            nn.Linear(16, dim * 1), nn.ReLU()
        ))
        self.normalize_out = tg.nn.norm.InstanceNorm(dim)
        self.edge_conv_out = smallEdgeModel(4, dim, 1)

        # activations
        self.node_activation = node_activation
        self.edge_activation = edge_activation

    def forward(self, D):
        x, edge_index, edge_attr = D.x, D.edge_index, D.edge_attr
        x = x.reshape((-1, 1))

        row = edge_index[0]
        col = edge_index[1]

        # conv 1
        x = nnF.relu(self.node_conv_in(self.normalize_in(x), edge_index, edge_attr)) + x
        edge_attr = nnF.relu(self.edge_conv_in(x[row], x[col], edge_attr.float())) + edge_attr

        for i in range(self.num_internal_conv):
            x = nnF.relu(self.node_convs[i](self.normalizations[i](x), edge_index, edge_attr)) + x
            edge_attr = nnF.relu(self.edge_convs[i](x[row], x[col], edge_attr.float())) + edge_attr

        # conv 6
        x = self.node_activation(self.node_conv_out(self.normalize_out(x), edge_index, edge_attr))
        edge_attr = self.edge_activation(self.edge_conv_out(x[row], x[col], edge_attr.float()))

        return x, edge_attr

class PNet(nn.Module):
    def __init__(self, dim, device):
        super(PNet, self).__init__()

        self.P_values = MPNN(dim)
        self.device = device
        self.to(device)

    def forward_P_hat(self, data):
        nodes, edges = self.P_values.forward(data)
        P_hat_vals = edges.squeeze()
        P_hat = torch.sparse_coo_tensor(data.edge_index, P_hat_vals, (data.x.shape[0], data.x.shape[0])).to(self.device)

        return P_hat.coalesce()

    def forward(self, data, agg_csr):
        agg_T = sparse.to_torch_sparse(agg_csr).to(self.device)

        # First, compute \hat{P}, or some matrix w/ same sparsity as A
        nodes, edges = self.P_values.forward(data)
        P_hat_vals = edges.squeeze()
        P_hat = torch.sparse_coo_tensor(data.edge_index, P_hat_vals, (data.x.shape[0], data.x.shape[0])).to(self.device)

        # Compute P := \hat{P} \hat{Agg}
        return torch.sparse.mm(P_hat, agg_T).coalesce()

class AggNet(nn.Module):
    def __init__(self, node_activation=nn.ReLU(), edge_activation=nn.ReLU()):
        super(AggNet, self).__init__()

        self.norm1 = tg.nn.norm.InstanceNorm(1);  self.nc1 = tg.nn.ChebConv(1,  4, K=3)
        self.norm2 = tg.nn.norm.InstanceNorm(4);  self.nc2 = tg.nn.ChebConv(4,  16, K=3)
        self.norm3 = tg.nn.norm.InstanceNorm(16); self.nc3 = tg.nn.ChebConv(16, 64, K=3)
        self.norm4 = tg.nn.norm.InstanceNorm(64); self.nc4 = tg.nn.ChebConv(64, 1, K=3)
        self.ec1 = smallEdgeModel(1*2+1, 2, 1)
        self.ec2 = smallEdgeModel(1*2+1, 2, 1)

        self.lin1 = nn.Linear(85, 40)
        self.lin2 = nn.Linear(40, 16)
        self.lin3 = nn.Linear(16, 1)

        self.node_activation = node_activation
        self.edge_activation = edge_activation

    def forward(self, D):
        x, edge_index, edge_attr = D.x, D.edge_index, D.edge_attr
        x = x.reshape((-1, 1))

        row = edge_index[0]
        col = edge_index[1]

        x1 = nnF.relu(self.nc1(self.norm1(x), edge_index, edge_attr))
        x2 = nnF.relu(self.nc2(self.norm2(x1), edge_index, edge_attr))
        x3 = nnF.relu(self.nc3(self.norm3(x2), edge_index, edge_attr))
        x4 = nnF.relu(self.nc4(self.norm4(x3), edge_index, edge_attr))

        x_stack = torch.column_stack((x1, x2, x3, x4))
        x = nnF.relu(self.lin1(x_stack))
        x = nnF.relu(self.lin2(x))
        x = self.node_activation(self.lin3(x))

        edge_attr = nnF.relu(self.ec1(x[row], x[col], edge_attr.float()))
        edge_attr = self.edge_activation(self.ec2(x[row], x[col], edge_attr.float()))

        return x, edge_attr

class FullAggNet(nn.Module):
    '''
    A model that outputs an interpolation operator for a matrix by internally
    forming aggregates, then smoothing those aggregates.

    Note that this currently does not pass gradients into the final interpolation
    operator, so some gradient-free optimizer is needed to train such as particle
    swarm or genetic algorithms.
    '''

    def __init__(self, dim=64, use_pnet=True, use_aggnet=True):
        super(FullAggNet, self).__init__()

        if use_pnet:
            self.PNet = MPNN(dim, num_internal_conv=4)
        else:
            self.PNet = None

        if use_aggnet:
            self.AggNet = AggNet()
        else:
            self.AggNet = None

    def forward_fixed_agg(self, A, agg):
        m, n = A.shape

        if isinstance(agg, sp.spmatrix):
            agg_T = ns.lib.sparse.to_torch_sparse(agg)
            agg_sp = agg
        else:
            agg_T = agg
            agg_sp = ns.lib.sparse_tensor.to_scipy(agg)

        data = ns.model.data.graph_from_matrix(A, agg_sp)
        nodes, edges = self.PNet(data)
        P_hat_vals = edges.squeeze()
        P_hat = torch.sparse_coo_tensor(data.edge_index, P_hat_vals, (m, n)).coalesce()

        return torch.sparse.mm(P_hat, agg_T).coalesce()

    def forward(self, A, alpha):
        '''
        Parameters
        ----------
        A : scipy.sparse.csr_matrix
          System to compute interpolation on
        alpha : float
          Ratio of nodes to use as aggregate centers.
          This coarsens the fine grid by roughly 1/alpha

        Returns
        -------
        agg : torch.sparse_coo_tensor
          The aggregate assignment matrix
        P : torch.sparse.coo_tensor
          Interpolation operator, mapping from the coarse grid to fine grid.
          The transpose will map from the fine grid to the coarse grid.
        bf_weights : torch.sparse_coo_tensor
          The weights matrix used for Bellman-Ford
        cluster_centers : torch.Tensor
          Centers of each cluster s.t. cluster_centers[i] is the root node of cluster i
        node_weights : torch.Tensor
          Value of each node when scoring is applied to pick centers.
        '''

        m, n = A.shape
        k = int(np.ceil(alpha * m))

        # Compute node scores and edge weights
        data_simple = ns.model.data.graph_from_matrix_basic(A)
        node_scores, edges = self.AggNet(data_simple)
        node_scores = node_scores.squeeze(); edges = edges.squeeze()
        BF_weights = torch.sparse_coo_tensor(data_simple.edge_index, edges, (m, n)).coalesce()

        # Find cluster centers
        top_k = torch.argsort(node_scores, descending=True)[:k]

        # Run Bellman-Ford to assign each node to an aggregate
        distance, nearest_center = ns.lib.graph.modified_bellman_ford(BF_weights, top_k)
        agg_T = ns.lib.graph.nearest_center_to_agg(top_k, nearest_center)

        if self.PNet:
            # Compute the smoother \hat{P}
            data = ns.model.data.graph_from_matrix(A, ns.lib.sparse_tensor.to_scipy(agg_T.detach()))
            nodes, edges = self.PNet(data)
            P_hat_vals = edges.squeeze()
            P_hat = torch.sparse_coo_tensor(data.edge_index, P_hat_vals, (m, n)).coalesce()
            # Now, form P := \hat{P} Agg.
            P = torch.sparse.mm(P_hat, agg_T).coalesce()
        else:
            P = None

        return agg_T, P, BF_weights, top_k, node_scores
