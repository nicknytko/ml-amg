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

        # Find the optimal column re-weightings for Agg, and compute \hat{Agg}
        # agg_weight_values = self.P_column_weights(agg_T.values(), network_data.edge_index, network_data.edge_attr).squeeze()
        # agg_weighted = torch.sparse_coo_tensor(agg_T.indices(), agg_weight_values, agg_T.size()).to(self.device)

        # Compute P := \hat{P} \hat{Agg}
        return torch.sparse.mm(P_hat, agg_T).coalesce()

class FullAggNet(nn.Module):
    '''
    A model that outputs an interpolation operator for a matrix by internally
    forming aggregates, then smoothing those aggregates.

    Note that this currently does not pass gradients into the final interpolation
    operator, so some gradient-free optimizer is needed to train such as particle
    swarm or genetic algorithms.
    '''

    def __init__(self, dim=64):
        super(FullAggNet, self).__init__()

        self.PNet = MPNN(dim//4, num_internal_conv=4)
        self.AggNet = MPNN(dim, node_activation=nn.Identity(), num_internal_conv=1)

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
        '''

        m, n = A.shape
        k = int(np.ceil(alpha * m))

        # Compute aggregates
        data_simple = ns.model.data.graph_from_matrix_basic(A)
        nodes, edges = self.AggNet(data_simple)
        nodes = nodes.squeeze(); edges = edges.squeeze()
        BF_weights = torch.sparse_coo_tensor(data_simple.edge_index, edges, (m, n)).coalesce()

        # Find cluster centers
        top_k = torch.argsort(nodes)[:k]

        # Run Bellman-Ford to assign each node to an aggregate
        distance, nearest_center = ns.lib.graph.modified_bellman_ford(BF_weights, top_k)
        agg_T = ns.lib.graph.nearest_center_to_agg(top_k, nearest_center)

        # Compute the smoother \hat{P}
        data = ns.model.data.graph_from_matrix(A, ns.lib.sparse_tensor.to_scipy(agg_T.detach()))
        nodes, edges = self.PNet(data)
        P_hat_vals = edges.squeeze()
        P_hat = torch.sparse_coo_tensor(data.edge_index, P_hat_vals, (m, n)).coalesce()

        # Now, form P := \hat{P} Agg.
        return agg_T, torch.sparse.mm(P_hat, agg_T).coalesce(), BF_weights, top_k
