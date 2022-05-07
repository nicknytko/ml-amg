import torch
import pyamg.graph
import numpy as np
import scipy.sparse as sp


def modified_bellman_ford(S_T, centers):
    '''
    Adapted from Algorithm 6.2 from Algebraic Multigrid for Discrete Differential Forms,
    Ph.D thesis, Nathan Bell
    (http://wnbell.com/blog/2008/08/01/algebraic-multigrid-for-discrete-differential-forms/)

    Parameters
    ----------
    S_T : torch.sparse_coo_tensor
      strength of connection matrix
    centers : torch.tensor
      length #clusters 1d integer tensor containing indices of cluster centers

    Returns
    -------
    distance : torch.tensor
      distance from each node to cluster center
    nearest_center : torch.tensor
      nearest cluster assignment for each node
    '''

    n = S_T.size()[0]

    distance = torch.ones(n) * float('inf')
    nearest_center = torch.zeros(n, dtype=torch.long)

    for c in centers:
        distance[c] = 0
        nearest_center[c] = c

    nonzero_indices = S_T.indices().T
    nonzero_values = S_T.values()

    while True:
        finished = True

        for idx, (i,j) in enumerate(nonzero_indices):
            dij = nonzero_values[idx]
            if distance[i] + dij < distance[j]:
                distance[j] = distance[i] + dij
                nearest_center[j] = nearest_center[i]
                finished = False

        if finished:
            break

    return distance.to(centers.device), nearest_center.to(centers.device)


def nearest_center_to_agg(top_k, nearest_center):
    '''
    Computes the aggregate matrix given a list of cluster assignments.

    Parameters
    ----------
    top_k : torch.tensor
      length #clusters 1d integer tensor containing the cluster centers
    nearest_center : torch.tensor
      length n cluster assignment for each node

    Returns:
    agg : torch.sparse_coo_tensor
      size (#clusters, n) aggregate assignment matrix
    '''

    n = len(nearest_center)
    m = len(top_k)

    # map cluster center -> column number
    inv_top_k = {}
    for i, k in enumerate(top_k):
        inv_top_k[k.item()] = i

    # generate agg matrix
    indices = []
    for i, c in enumerate(nearest_center):
        k_i = inv_top_k[c.item()]
        indices.append([i, k_i])

    return torch.sparse_coo_tensor(torch.Tensor(indices).T, torch.ones(n), (n, m), device=top_k.device).coalesce()


def num_connected_components(adj):
    '''
    Returns the number of connected components in some graph, given its adjacency matrix.

    Parameters
    ----------
    adj : numpy.ndarray or scipy.sparse_matrix
      Adjacency matrix of graph to compute connected components for

    Returns
    -------
    num_components : integer
      Number of connected components, where node u,v are connected for all u,v in the component
    '''

    n = adj.shape[0]
    visited = np.zeros(n, dtype=bool)

    def first_unvisited(visited):
        return np.where(visited == False)[0][0]

    num_bfs = 0
    while not np.all(visited):
        num_bfs += 1
        stack = [first_unvisited(visited)]
        while len(stack) > 0:
            i = stack.pop()
            visited[i] = True
            neighbours = np.nonzero(adj[:,i])[0]
            for n in neighbours:
                if not visited[n]:
                    stack.append(n)

    return num_bfs


def check_aggregates_connected(A, Agg):
    '''
    Checks that all aggregates in a tentative aggregation are composed of connected nodes.
    I.e., nodes u,v in aggregate j can be reached for all u, v, j.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
      Connection matrix
    Agg : scipy.sparse.csr_matrix
      Tentative aggregate assignments

    Returns
    -------
    is_connected: bool
      If all aggregates are fully-connected.
    '''

    # Form A' with inter-aggregate connections removed
    n, k = Agg.shape
    Ap = []
    for i in range(k):
        R_i = sp.eye(n).tocsc()[:, Agg[:,i].nonzero()[0]]
        Ap.append(R_i.T@A@R_i)
    Ap = sp.block_diag(Ap)

    # If all aggregates are connected, then the number of overall connected components in the graph
    # should be equal to the number of aggregates
    return (num_connected_components(Ap.tocsc()) == k)


def lloyd_aggregation(C, ratio=0.03, distance='unit', maxiter=10, rand=None):
    """
    Aggregate nodes using Lloyd Clustering.

    Stolen/modified from PyAMG:
      https://github.com/pyamg/pyamg/blob/e3fb6feaad2358e681f2f4affae3205bfe9a2350/pyamg/aggregation/aggregate.py#L179

    Parameters
    ----------
    C : csr_matrix
        strength of connection matrix
    ratio : scalar
        Fraction of the nodes which will be seeds.
    distance : ['unit','abs','inv',None]
        Distance assigned to each edge of the graph G used in Lloyd clustering
        For each nonzero value C[i,j]:
        =======  ===========================
        'unit'   G[i,j] = 1
        'abs'    G[i,j] = abs(C[i,j])
        'inv'    G[i,j] = 1.0/abs(C[i,j])
        'same'   G[i,j] = C[i,j]
        'sub'    G[i,j] = C[i,j] - min(C)
        =======  ===========================
    maxiter : int
        Maximum number of iterations to perform
    rand : np.random.RandomState
        Random state to use when generating seed points

    Returns
    -------
    AggOp : csr_matrix
        aggregation operator which determines the sparsity pattern
        of the tentative prolongator
    roots : array
        array of Cpts, i.e., Cpts[i] = root node of aggregate i
    seeds : array
        array of initial roots/seed points for Lloyd, following same structure as roots
    """
    if ratio <= 0 or ratio > 1:
        raise ValueError('ratio must be > 0.0 and <= 1.0')

    if not (sp.isspmatrix_csr(C) or sp.isspmatrix_csc(C)):
        raise TypeError('expected csr_matrix or csc_matrix')

    # Distance metric
    if distance == 'unit':
        data = np.ones_like(C.data).astype(float)
    elif distance == 'abs':
        data = abs(C.data)
    elif distance == 'inv':
        data = 1.0/abs(C.data)
    elif distance == 'same':
        data = C.data
    elif distance == 'min':
        data = C.data - C.data.min()
    else:
        raise ValueError(f'Unrecognized value distance={distance}')

    # Random state
    if rand is None:
        rand = np.random
    elif isinstance(rand, int):
        rand = np.random.RandomState(rand)
    elif not isinstance(rand, np.random.RandomState):
        raise TypeError('rand should be an integer seed value or a random state')

    if C.dtype == complex:
        data = np.real(data)

    assert data.min() >= 0

    G = C.__class__((data, C.indices, C.indptr), shape=C.shape)

    N = C.shape[0]
    num_seeds = int(np.ceil(ratio * N))
    seeds = rand.permutation(N)[:num_seeds]
    _, clusters, roots = pyamg.graph.lloyd_cluster(G, np.copy(seeds), maxiter=maxiter)

    row = (clusters >= 0).nonzero()[0]
    col = clusters[row]
    data = np.ones(len(row), dtype='int8')
    AggOp = sp.coo_matrix((data, (row, col)),
                          shape=(G.shape[0], num_seeds)).tocsr()
    return AggOp, roots, seeds
