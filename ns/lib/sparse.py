import torch
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import numpy as np

def col_normalize_csr(A_sp, ord=1):
    '''
    Normalizes the columns of a sparse CSR matrix.
    '''
    norms = spla.norm(A_sp, axis=0, ord=ord)
    new_data = A_sp.data / norms[A_sp.indices] # divide each nonzero by its respective column norm
    return sp.csr_matrix((new_data, A_sp.indices, A_sp.indptr), A_sp.shape)

def to_torch_sparse(A):
    A = A.tocoo()
    A_T = torch.sparse_coo_tensor(
        torch.Tensor(np.row_stack([A.row, A.col])),
        torch.Tensor(A.data),
        A.shape
    )
    return A_T.coalesce()