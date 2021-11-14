import networkx as nx
import torch
import torch_geometric as tg
import numpy as np

def graph_from_matrix(A, agg_op):
    n = A.shape[0]

    G = nx.from_scipy_sparse_matrix(A, edge_attribute='weight', parallel_edges=False, create_using=nx.DiGraph)
    x = torch.ones(n) / n

    # Create cluster feature
    clusters = np.array(agg_op.argmax(axis=1)).flatten()
    cluster_adj = {} # 0 if in same cluster, 1 if not
    for (u, v) in nx.edges(G):
        adj = (0 if (clusters[u] == clusters[v]) else 1)
        cluster_adj[(u, v)] = adj

    nx.set_edge_attributes(G, cluster_adj, 'cluster_adj')
    nx_data = tg.utils.from_networkx(G, None, ['weight', 'cluster_adj'])
    return tg.data.Data(x=x, edge_index=nx_data.edge_index, edge_attr=abs(nx_data.edge_attr.float()))

def graph_from_matrix_basic(A):
    n = A.shape[0]

    G = nx.from_scipy_sparse_matrix(A, edge_attribute='weight', parallel_edges=False, create_using=nx.DiGraph)
    x = torch.ones(n) / n

    # Create cluster feature
    cluster_adj = {} # 0 if in same cluster, 1 if not
    for (u, v) in nx.edges(G):
        cluster_adj[(u, v)] = 1.0 / n

    nx.set_edge_attributes(G, cluster_adj, 'cluster_adj')
    nx_data = tg.utils.from_networkx(G, None, ['weight', 'cluster_adj'])
    return tg.data.Data(x=x, edge_index=nx_data.edge_index, edge_attr=abs(nx_data.edge_attr.float()))
