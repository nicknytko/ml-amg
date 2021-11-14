import torch

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

    return distance, nearest_center

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

    return torch.sparse_coo_tensor(torch.Tensor(indices).T, torch.ones(n), (n, m)).coalesce()
