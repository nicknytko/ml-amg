import torch
import torch.nn as nn
import torch.linalg as tla
import torch.sparse
import torch_sparse as ts
import numpy as np
import scipy.sparse as sp

def spspmm(A, B):
    '''
    Sparse * Sparse mat-mat
    '''
    assert(A.shape[1] == B.shape[0])

    m, k = A.shape
    n = B.shape[1]

    i, v = ts.spspmm(A.indices(), A.values(), B.indices(), B.values(), m, k, n, False)
    T = torch.sparse_coo_tensor(i, v, (m, n))
    return T.coalesce()

def spmm(A, B):
    '''
    Sparse * Dense mat-mat
    '''
    assert(A.shape[1] == B.shape[0])

    m, n = A.shape
    return ts.spmm(A.indices(), A.values(), m, n, B)

def spT(A):
    '''
    Sparse transpose
    '''
    m, n = A.shape
    i, v = ts.transpose(A.indices(), A.values(), m, n)
    T = torch.sparse_coo_tensor(i, v, (n, m))
    return T.coalesce()

def diag(A):
    '''
    Get the diagonal of a sparse tensor
    '''

    n = min(A.shape[0], A.shape[1])
    d = torch.ones(n)
    indices, values = (A.indices(), A.values())
    for i in range(indices.shape[1]):
        j, k = indices[:,i]
        if j == k:
            d[j] = values[i]
    return d

def to_scipy(T):
    indices = np.array(T.indices())
    coo =  sp.coo_matrix((np.array(T.values()),
                         (indices[0], indices[1])),
                         shape=np.array(T.shape))
    return coo.tocsr()
