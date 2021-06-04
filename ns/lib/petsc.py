from firedrake import *
from firedrake.petsc import PETSc
import scipy.sparse as sp

def petsc_to_csr(P):
    '''
    Convert a petsc4py matrix into an equivalent
    scipy sparse CSR matrix.
    
    csr = petsc_to_csr(petsc)

    P - petsc4py Mat object
    '''
    
    row, col, val = P.getValuesCSR()
    return sp.csr_matrix((val, col, row))
