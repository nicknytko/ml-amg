import torch
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import numpy as np
import ns.lib.sparse_tensor


def col_normalize_csr(A_sp, ord=1):
    '''
    Normalizes the columns of a sparse CSR matrix.
    '''
    if not sp.isspmatrix_csr(A_sp):
        A_sp = A_sp.tocsr()

    norms = spla.norm(A_sp, axis=0, ord=ord)
    new_data = A_sp.data / norms[A_sp.indices] # divide each nonzero by its respective column norm
    return sp.csr_matrix((new_data, A_sp.indices, A_sp.indptr), A_sp.shape)


def to_torch_sparse(A):
    '''
    Convert a scipy sparse matrix into a torch sparse COO tensor
    '''

    A = A.tocoo()
    A_T = torch.sparse_coo_tensor(
        torch.Tensor(np.row_stack([A.row, A.col])),
        torch.Tensor(A.data),
        A.shape
    )
    A_T = A_T.coalesce()
    return A_T


def get_diagonal(A_T, as_vector=True):
    '''
    Extracts the diagonal from a torch sparse COO tensor
    '''

    values = A_T.values()
    indices = A_T.indices()
    diag_entries = (indices[0] == indices[1])

    if as_vector:
        n = min(A_T.shape[0], A_T.shape[1])
        return values[diag_entries]
    else:
        return torch.sparse_coo_tensor(indices[diag_entries], values[diag_entries], size=A_T.shape)


def triu(A_T, diag=0):
    '''
    Extracts the upper triangular portion of a sparse COO tensor.

    Parameters
    ----------
    A_T : torch.sparse_coo_tensor
      Sparse tensor to extract triangular portion from
    diag : integer
      Which diagonal to begin from.  Values are indicated by:
      < 0: below main diagonal
      = 0: main diagonal
      > 0: above main diagonal

    Returns
    -------
    U_T : torch.sparse_coo_tensor
      New tensor containing upper triangular entries of A_T
    '''

    values = A_T.values()
    indices = A_T.indices()
    entry_mask = (indices[1] - indices[0]) >= diag

    return torch.sparse_coo_tensor(indices[:, entry_mask], values[entry_mask], size=A_T.shape)


def tril(A_T, diag=0):
    '''
    Extracts the lower triangular portion of a sparse COO tensor.

    Parameters
    ----------
    A_T : torch.sparse_coo_tensor
      Sparse tensor to extract triangular portion from
    diag : integer
      Which diagonal to begin from.  Values are indicated by:
      < 0: below main diagonal
      = 0: main diagonal
      > 0: above main diagonal

    Returns
    -------
    U_T : torch.sparse_coo_tensor
      New tensor containing lower triangular entries of A_T
    '''

    values = A_T.values()
    indices = A_T.indices()
    entry_mask = (indices[0] - indices[1]) >= diag

    return torch.sparse_coo_tensor(indices[:, entry_mask], values[entry_mask], size=A_T.shape)


scipy_to_torch = to_torch_sparse
torch_to_scipy = ns.lib.sparse_tensor.to_scipy
