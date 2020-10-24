# qr_decomposition.py
"""Volume 1: The QR Decomposition.
Caelan Osman
Math 345 Sec 3
October 23, 2020
"""

import numpy as np
from scipy import linalg as la

# Problem 1
def qr_gram_schmidt(A):
    """ This function computes the reduced QR decomposition of A via Modified Gram-Schmidt.
        Note: The matrix A (m x n) has to have rank n which is less than or equaal to m.
    """
    n = A.shape[1] #get the size that will help form Q and R
    Q = np.copy(A).astype(np.float64) #copy A and change type to float
    R = np.zeros((n,n)) #intialize R with an nxn matrix of zeros
    for i in range(0, n):
        R[i, i] = la.norm(Q[:, i]) #diagonals will be set to the norm of the ith column
        Q[:, i] /= R[i, i] #normalize the columns of Q
        for j in range (i+1, n):
            R[i, j] = (Q[:, j]).T @ (Q[:, i]) #make the upper triangluar elements of R
            Q[:, j] -= (R[i, j] * Q[:, i]) #orthognoalize the jth column of Q
    return Q, R

# Problem 2
def abs_det(A):
    """Use the QR decomposition to efficiently compute the absolute value of
    the determinant of A. A needs to be invertible
    """
    return abs(np.prod(np.diag(la.qr(A)[1])))

# Problem 3
def solve(A, b):
    """Use the QR decomposition to efficiently solve the system Ax = b.
    """




# Problem 4
def qr_householder(A):
    """Compute the full QR decomposition of A via Householder reflections.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,m) ndarray): An orthonormal matrix.
        R ((m,n) ndarray): An upper triangular matrix.
    """
    raise NotImplementedError("Problem 4 Incomplete")


# Problem 5
def hessenberg(A):
    """Compute the Hessenberg form H of A, along with the orthonormal matrix Q
    such that A = QHQ^T.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.

    Returns:
        H ((n,n) ndarray): The upper Hessenberg form of A.
        Q ((n,n) ndarray): An orthonormal matrix.
    """
    raise NotImplementedError("Problem 5 Incomplete")
