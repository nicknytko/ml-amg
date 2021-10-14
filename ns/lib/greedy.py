import numpy as np


def diag_dominance(A):
    """Returns the measures of the diagonal dominance for each row of a matrix"""
    no_rows = A.shape[0]
    dominance = np.zeros(no_rows)
    for i in range(no_rows):
        dominance[i] = np.absolute(A[i,i]) / np.sum(np.absolute(A[i,:]))
    return dominance


def greedy_coarsening(A, theta):
    """Returns the set of fine and coarse grid points and number of fine grid points"""
    dominance = diag_dominance(A)
    length = len(dominance)
    F = np.array([], dtype=int)
    U = np.arange(length, dtype=int)
    C = np.array([], dtype=int)
    for i in range(length):
        if dominance[i] >= theta:
            F = np.append(F, i)
    U = np.setdiff1d(U, F)
    while len(U) >= 1:
        c = np.argmin(dominance[U])
        C = np.append(C,U[c])
        l = np.nonzero(A[U[c],:])[1]
        U = np.setdiff1d(U, U[c])
        for i in np.intersect1d(U, l):
            sum1 = np.sum(np.absolute(A[i,U]))
            sum2 = np.sum(np.absolute(A[i,F]))
            dominance[i] = np.absolute(A[i,i]) / (sum1 + sum2)
            if dominance[i] >= theta:
                F = np.append(F, i)
                U = np.setdiff1d(U, F)
    return len(F), F, C


def greedy_coarsening_opt(A, theta):
    """Returns the set of fine and coarse grid points and number of fine grid points"""
    dominance = diag_dominance(A)
    length = len(dominance)
    F = set()
    C = set()
    U = set(np.arange(length, dtype=int))
    for i in range(length):
        if dominance[i] >= theta:
            #F = np.append(F, i)
            F.add(i)
    #U = np.setdiff1d(U, F)
    U = U.difference(F)
    while len(U) >= 1:
        c = np.argmin(dominance[np.array(U)])
        #C = np.append(C,U[c])
        C.add(U[c])
        l = np.nonzero(A[U[c],:])[1]
        U = np.setdiff1d(U, U[c])
        for i in np.intersect1d(U, l):
            sum1 = np.sum(np.absolute(A[i,U]))
            sum2 = np.sum(np.absolute(A[i,F]))
            dominance[i] = np.absolute(A[i,i]) / (sum1 + sum2)
            if dominance[i] >= theta:
                F = np.append(F, i)
                U = np.setdiff1d(U, F)
    return len(F), F, C
