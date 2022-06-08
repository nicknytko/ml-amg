import torch
import torch_geometric as tg
import torch.nn as nn
import torch.nn.functional as nnF
import numpy as np
import scipy.sparse as sp
import ns.lib.sparse as sparse
import ns.model.data
import ns.lib.graph
import pyamg
from ns.lib.profiler import Profiler


def topk_vec(x, k):
    if len(x.shape) != 1:
        x = x.squeeze()
    assert(len(x.shape) == 1)

    top_k = torch.argsort(x, descending=True)[:k]
    top_k_vec = torch.zeros(x.shape, device=x.device)
    top_k_vec[top_k] = 1.0 # (n, 1), with 1.0 for cluster centers and 0.0 elsewhere
    return top_k_vec


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


class EdgeConvModel(nn.Module):
    def __init__(self, node_dim, in_edge_dim, out_edge_dim, hid_dim=16):
        super(EdgeConvModel, self).__init__()
        self.edge_mlp = nn.Sequential(nn.Linear(node_dim*2+in_edge_dim, hid_dim), nn.ReLU(),
                                      nn.LayerNorm([hid_dim]),
                                      nn.Linear(hid_dim, hid_dim), nn.ReLU(),
                                      nn.LayerNorm([hid_dim]),
                                      nn.Linear(hid_dim, out_edge_dim))

    def forward(self, x, edge_index, edge_attr):
        if len(x.shape) == 1:
            x = torch.unsqueeze(x, 1)

        frm = edge_index[0]
        to = edge_index[1]

        inpt = torch.cat([x[frm], x[to], edge_attr.float()], 1)
        out = self.edge_mlp(inpt)
        return out


class MPNN(nn.Module):
    def __init__(self, dim, node_activation=nn.ReLU(), edge_activation=nn.ReLU(), num_internal_conv=4, input_edge_features=1):
        super(MPNN, self).__init__()

        # input -> hidden dim layer
        self.node_conv_in = tg.nn.NNConv(1, dim, nn=nn.Sequential(
            TensorLambda(lambda x: x.reshape(-1, input_edge_features)),
            nn.Linear(input_edge_features, 4), nn.ReLU(),
            nn.Linear(4, 16), nn.ReLU(),
            nn.Linear(16, 1 * dim), nn.ReLU()
        ))
        self.normalize_in = tg.nn.norm.InstanceNorm(dim)
        self.edge_conv_in = smallEdgeModel(dim*2+input_edge_features, dim, 2)

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

        with Profiler('input convolution'):
            # input convolution
            x = nnF.relu(self.node_conv_in(self.normalize_in(x), edge_index, abs(edge_attr))) + x
            edge_attr = nnF.relu(self.edge_conv_in(x[row], x[col], edge_attr.float())) + edge_attr

        # middle convolutions
        for i in range(self.num_internal_conv):
            with Profiler(f'convolution {i}'):
                with Profiler('node convolution'):
                    x = nnF.relu(self.node_convs[i](self.normalizations[i](x), edge_index, edge_attr)) + x
                with Profiler('edge convolution'):
                    edge_attr = nnF.relu(self.edge_convs[i](x[row], x[col], edge_attr.float())) + edge_attr

        with Profiler('output convolution'):
            # output convolution
            x = self.node_activation(self.node_conv_out(self.normalize_out(x), edge_index, edge_attr))
            edge_attr = self.edge_activation(self.edge_conv_out(x[row], x[col], edge_attr.float()))

        return x, edge_attr


class AggBinarizationLayer(nn.Module):
    def __init__(self, dim, num_conv=6):
        super(AggBinarizationLayer, self).__init__()
        ncs = []
        fcs = []
        norms = []

        # Input -> Hidden
        ncs.append(tg.nn.TAGConv(1, dim))
        fcs.append(
            nn.Sequential(
                nn.Linear(dim, dim), nn.ReLU(),
                nn.Linear(dim, dim), nn.ReLU(),
                nn.Linear(dim, dim), nn.ReLU(),
                nn.Linear(dim, dim), nn.ReLU(),
                nn.Linear(dim, dim), nn.ReLU()
            )
        )
        norms.append(tg.nn.norm.InstanceNorm(1))

        # Hidden -> Hidden
        for i in range(num_conv-2):
            ncs.append(tg.nn.TAGConv(dim, dim))
            fcs.append(
                nn.Sequential(
                    nn.Linear(dim, dim), nn.ReLU(),
                    nn.Linear(dim, dim), nn.ReLU(),
                    nn.Linear(dim, dim), nn.ReLU(),
                    nn.Linear(dim, dim), nn.ReLU(),
                    nn.Linear(dim, dim), nn.ReLU()
                )
            )
            norms.append(tg.nn.norm.InstanceNorm(dim))

        # Hidden -> Output
        ncs.append(tg.nn.TAGConv(dim, dim))
        fcs.append(
            nn.Sequential(
                nn.Linear(dim, dim), nn.ReLU(),
                nn.Linear(dim, dim), nn.ReLU(),
                nn.Linear(dim, dim), nn.ReLU(),
                nn.Linear(dim, dim), nn.ReLU(),
                nn.Linear(dim, 1), nn.ReLU()
            )
        )
        norms.append(tg.nn.norm.InstanceNorm(dim))

        self.ncs = nn.ModuleList(ncs)
        self.fcs = nn.ModuleList(fcs)
        self.norms = nn.ModuleList(norms)
        self.num_conv = num_conv
        self.dim = dim


    def forward(self, x, edge_index, edge_attr, k):
        if len(x.shape) == 1:
            x = torch.unsqueeze(x, 1)

        for i in range(self.num_conv):
            x = self.norms[i](x)
            x = self.ncs[i](x, edge_index, edge_attr)
            x = nnF.relu(x)
            x = self.fcs[i](x)

        top_k_vec = topk_vec(x, k)
        return top_k_vec, edge_attr


    def forward_raw(self, x, edge_index, edge_attr):
        if len(x.shape) == 1:
            x = torch.unsqueeze(x, 1)
        n = x.shape[0]

        for i in range(self.num_conv):
            x = self.norms[i](x)
            x = self.ncs[i](x, edge_index, edge_attr)
            x = nnF.relu(x)
            x = self.fcs[i](x)

        return x


class AggNet(nn.Module):
    def __init__(self, dim, iterations=2, num_conv=6):
        super(AggNet, self).__init__()
        layers = []
        for i in range(iterations):
            layers.append(AggBinarizationLayer(dim, num_conv=num_conv))
        self.layers = nn.ModuleList(layers)
        self.num_iterations = iterations

    def forward(self, D, k):
        x, edge_index, edge_attr = D.x, D.edge_index, D.edge_attr
        for i, layer in enumerate(self.layers):
            x, edge_attr = layer(x, edge_index, edge_attr, k)
        return x

    def all_intermediate_topk(self, D, k):
        x, edge_index, edge_attr = D.x, D.edge_index, D.edge_attr
        intermediate = []
        for i, layer in enumerate(self.layers):
            x, edge_attr = layer(x, edge_index, edge_attr, k)
            intermediate.append(torch.clone(x))
        return intermediate


class AggOnlyNet(nn.Module):
    def __init__(self, dim=64, num_conv=6, iterations=2):
        super(AggOnlyNet, self).__init__()
        self.AggNet = AggNet(dim, num_conv=num_conv, iterations=iterations)
        self.CNet = MPNN(dim, num_internal_conv=5)

    def forward(self, A, alpha):
        m, n = A.shape
        k = int(np.ceil(alpha * m))

        # Compute node scores
        data_simple = ns.model.data.graph_from_matrix_basic(A)
        node_scores = self.AggNet(data_simple, k).squeeze()
        top_k = torch.where(node_scores == 1)[0]

        # Output Bellman-ford weights
        BF_nodes, BF_edges = self.CNet(data_simple)
        C_T = torch.sparse_coo_tensor(data_simple.edge_index, BF_edges.squeeze(), (m, n)).coalesce()
        C = ns.lib.sparse.torch_to_scipy(C_T)

        # Run Bellman-Ford to assign each node to an aggregate
        distance, nearest_center = ns.lib.graph.modified_bellman_ford(C_T, top_k)
        agg_T = ns.lib.graph.nearest_center_to_agg(top_k, nearest_center)
        agg = ns.lib.sparse_tensor.to_scipy(agg_T)

        # Compute final interpolation through smoothed aggregation
        P = ns.lib.multigrid.smoothed_aggregation_jacobi(A, agg)
        P_T = ns.lib.sparse.scipy_to_torch(P)

        return agg, P_T, C, top_k, node_scores

    def forward_intermediate_topk(self, A, alpha):
        m, n = A.shape
        k = int(np.ceil(alpha * m))
        data_simple = ns.model.data.graph_from_matrix_basic(A)
        data_node_score = tg.data.Data(x=data_simple.x, edge_index=data_simple.edge_index, edge_attr=data_simple.edge_attr)

        return self.AggNet.all_intermediate_topk(data_node_score, k)


class SocNet(nn.Module):
    def __init__(self, dim, num_conv=4, input_edge_features=1):
        super(SocNet, self).__init__()
        assert(num_conv >= 2)

        ncs = []
        ecs = []
        fcs = []
        norms = []

        # Input -> Hidden
        norms.append(tg.nn.norm.InstanceNorm(1))
        ncs.append(tg.nn.TAGConv(1, dim))
        fcs.append(
            nn.Sequential(
                nn.Linear(dim, dim), nn.ReLU(),
                nn.Linear(dim, dim), nn.ReLU(),
                nn.Linear(dim, dim), nn.ReLU(),
                nn.Linear(dim, dim), nn.ReLU()
            )
        )
        ecs.append(EdgeConvModel(dim, 1, 1, dim))

        # Hidden -> Hidden
        for i in range(num_conv-2):
            norms.append(tg.nn.norm.InstanceNorm(dim))
            ncs.append(tg.nn.TAGConv(dim, dim))
            fcs.append(
                nn.Sequential(
                    nn.Linear(dim, dim), nn.ReLU(),
                    nn.Linear(dim, dim), nn.ReLU(),
                    nn.Linear(dim, dim), nn.ReLU(),
                    nn.Linear(dim, dim), nn.ReLU()
                )
            )
            ecs.append(EdgeConvModel(dim, 1, 1, dim))

        # Hidden -> Output
        norms.append(tg.nn.norm.InstanceNorm(dim))
        ncs.append(tg.nn.TAGConv(dim, dim))
        fcs.append(
            nn.Sequential(
                nn.Linear(dim, dim), nn.ReLU(),
                nn.Linear(dim, dim), nn.ReLU(),
                nn.Linear(dim, dim), nn.ReLU(),
                nn.Linear(dim, 1), nn.ReLU()
            )
        )
        ecs.append(EdgeConvModel(1, 1, 1, dim))

        self.ncs = nn.ModuleList(ncs)
        self.fcs = nn.ModuleList(fcs)
        self.ecs = nn.ModuleList(ecs)
        self.norms = nn.ModuleList(norms)
        self.num_conv = num_conv
        self.dim = dim

    def forward(self, D):
        x, edge_index, edge_attr = D.x, D.edge_index, D.edge_attr
        if len(x.shape) == 1:
            x = torch.unsqueeze(x, 1)

        for i in range(self.num_conv):
            x = self.norms[i](x)
            x = self.ncs[i](x, edge_index, edge_attr)
            x = nnF.relu(x)
            x = self.fcs[i](x)
            edge_attr = nnF.relu(self.ecs[i](x, edge_index, edge_attr))

        return x, edge_attr


class FullAggNet(nn.Module):
    '''
    A model that outputs an interpolation operator for a matrix by internally
    forming aggregates, then smoothing those aggregates.

    Note that this currently does not pass gradients into the final interpolation
    operator, so some gradient-free optimizer is needed to train such as particle
    swarm or genetic algorithms.
    '''

    def __init__(self, dim=64, num_conv=2, iterations=4):
        super(FullAggNet, self).__init__()

        self.PNet = MPNN(dim, num_internal_conv=4, input_edge_features=2)
        self.AggNet = AggNet(dim, num_conv=num_conv, iterations=iterations)
        self.CNet = MPNN(dim, num_internal_conv=5)
        # self.CNet = SocNet(dim, num_conv=5)

    def forward_intermediate_topk(self, A, alpha):
        m, n = A.shape
        k = int(np.ceil(alpha * m))
        data_simple = ns.model.data.graph_from_matrix_basic(A)
        data_node_score = tg.data.Data(x=data_simple.x, edge_index=data_simple.edge_index, edge_attr=data_simple.edge_attr)

        return self.AggNet.all_intermediate_topk(data_node_score, k)

    def int_only(self, A, Agg):
        m, n = A.shape
        agg_T = ns.lib.sparse.scipy_to_torch(Agg)
        data = ns.model.data.graph_from_matrix(A, ns.lib.sparse_tensor.to_scipy(agg_T))
        nodes, edges = self.PNet(data)
        P_hat = torch.sparse_coo_tensor(data.edge_index, edges.squeeze(), (m, n)).coalesce()
        P_T = torch.sparse.mm(P_hat, agg_T).coalesce()
        return P_T

    def agg_only(self, A, alpha):
        m, n = A.shape
        k = int(np.ceil(alpha * m))
        data_simple = ns.model.data.graph_from_matrix_basic(A)

        # Compute node scores
        node_scores = self.AggNet(data_simple, k).squeeze()
        top_k = torch.where(node_scores == 1)[0]

        # Output Bellman-ford weights
        BF_nodes, BF_edges = self.CNet(data_simple)
        C_T = torch.sparse_coo_tensor(data_simple.edge_index, BF_edges.squeeze(), (m, n)).coalesce()

        # Run Bellman-Ford to assign each node to an aggregate
        distance, nearest_center = ns.lib.graph.modified_bellman_ford(C_T, top_k)
        agg_T = ns.lib.graph.nearest_center_to_agg(top_k, nearest_center)
        return agg_T

    def smoother(self, A, agg_T):
        m, n = A.shape
        data = ns.model.data.graph_from_matrix(A, ns.lib.sparse_tensor.to_scipy(agg_T.detach()))
        nodes, edges = self.PNet(data)
        return torch.sparse_coo_tensor(data.edge_index, edges.squeeze(), (m, n)).coalesce()

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, A, alpha):
        '''
        Parameters
        ----------
        A : scipy.sparse.csr_matrix
          System to compute interpolation and aggregates on
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
          The strength-of-connection matrix used for Bellman-Ford
        cluster_centers : torch.Tensor
          Centers of each cluster s.t. cluster_centers[i] is the root node of cluster i
        node_weights : torch.Tensor
          Value of each node when scoring is applied to pick centers.
        '''
        from time import time

        with Profiler('creating data'):
            m, n = A.shape
            k = int(np.ceil(alpha * m))
            data_simple = ns.model.data.graph_from_matrix_basic(A).to(self.device)

        with Profiler('computing node scores'):
            # Compute node scores
            node_scores = self.AggNet(data_simple, k).squeeze()
            top_k = torch.where(node_scores == 1)[0]

        with Profiler('computing bellman-ford weights'):
            # Output Bellman-ford weights
            BF_nodes, BF_edges = self.CNet(data_simple)
            #C_T = torch.sparse_coo_tensor(data_simple.edge_index, BF_edges.squeeze(), (m, n)).coalesce()
            edge_indices = data_simple.edge_index.cpu().numpy()
            edge_weights = BF_edges.squeeze().cpu().numpy()
            C = sp.coo_matrix((edge_weights, (edge_indices[0], edge_indices[1])), shape=(m, n))
            C_T = ns.lib.sparse.scipy_to_torch(C)

        with Profiler('running bellman-ford'):
            # Run Bellman-Ford to assign each node to an aggregate
            distance, nearest_center = pyamg.graph.bellman_ford(C, top_k.cpu().numpy())
            agg_T = ns.lib.graph.nearest_center_to_agg(top_k, nearest_center).to(self.device)

        with Profiler('computing smoother'):
            # Compute the smoother \hat{P}
            data = ns.model.data.graph_from_matrix(A, ns.lib.sparse_tensor.to_scipy(agg_T.detach())).to(self.device)
            nodes, edges = self.PNet(data)
            P_hat = torch.sparse_coo_tensor(data.edge_index, edges.squeeze(), (m, n), device=self.device).coalesce()

        with Profiler('smoothing aggregates'):
            # Now, form P := \hat{P} Agg.
            P_T = torch.sparse.mm(P_hat, agg_T).coalesce()
        # torch.cuda.synchronize()
        return agg_T, P_T, C_T, top_k, node_scores
