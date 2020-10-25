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
       We are assume that the matrix A is n x n and invertible
    """
    Q, R = la.qr(A) #compute QR decomposition
    y = Q.T @ b #calculate Y

    rows_x = R.shape[1]
    x = np.zeros(rows_x)
    for i in range(rows_x - 1, -1, -1): #start from the bottom go to the top
        x[i] = y[i] #setting initial element
        for j in range(i + 1, rows_x): #subtracting off previous x elements
            x[i] -= R[i, j] * x[j]
        x[i] /= R[i, i] #isolating x[i]

    return x

# Problem 4
def qr_householder(A):
    """Compute the full QR decomposition of A via Householder reflections.
    """
    sign = lambda x: 1 if x >= 0 else -1 #used for setting u

    m, n = A.shape #get shape of A
    R = np.copy(A).astype(np.float64) #copy and change t ype
    Q = np.eye(m) #create a m x m identity matrix
    for k in range(0, n):
        u = np.copy(R[k:, k]) #initialize u
        u[0] += sign(u[0]) * la.norm(u) #reassign the first element
        u /= la.norm(u) #normalize
        R[k:, k:] -= 2 * np.outer(u, np.dot(u, R[k:, k:])) #relfect R
        Q[k:, :] -= 2 * np.outer(u, (u.T @ Q[k:, :])) #reflect Q

    return Q.T, R

# Problem 5
def hessenberg(A):
    """Compute the Hessenberg form H of A, along with the orthonormal matrix Q
    such that A = QHQ^T. A has to be nonsingular
    """
    sign = lambda x: 1 if x >= 0 else -1 #used for setting u

    m, n = A.shape #get shape of A
    H = np.copy(A).astype(np.float64) #copy A and set type
    Q = np.eye(m) #create a mxm identity amtrix
    for k in range(0, n-2):
        u = np.copy(H[k+1:, k]) #initialize u
        u[0] += sign(u[0]) * la.norm(u) # reset the first element
        u /= la.norm(u) #normalize u
        H[k+1:, k:] -= 2 * np.outer(u, u.T @ H[k+1:, k:]) #apply Qk to H
        H[:, k+1:] -= 2 * np.outer(H[:, k+1:] @ u, u.T) #apply QkT to H
        Q[k+1:, :] -= 2 * np.outer(u, u.T @ Q[k+1:, :]) #apply Qk to Q

    return H, Q.T
