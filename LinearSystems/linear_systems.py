# linear_systems.py
"""Volume 1: Linear Systems.
Caelan Osman
Math 345 Sec 3
October 19, 2020
"""

import time
import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg as la
from scipy import sparse
from scipy.sparse import linalg as spla

# Problem 1
def ref(A):
    """This function reduces the square matrix A to REF.
    Parameters:
        A ((n,n) ndarray): The square invertible matrix to be reduced.
    Returns:
        ((n,n) ndarray): The REF of A.
    """
    A = A.astype(np.float64) #change type to float so we can modify
    rows, cols = A.shape[0], A.shape[1]
    for col in range(0, cols):
        for row in range(col + 1, rows):
            if A[col, col] == 0: #avoids divide by zero errors
                continue
            else: #setting our old row by subtracting a constant of the pivot row from the current row
                A[row, col:] -= (A[row, col]/A[col, col]) * A[col, col:]
    return A #return our ref matrix

# Problem 2
def lu(A):
    """This function computes the LU decomposition of the square matrix A.
    Parameters:
        A ((n,n) ndarray): The matrix to decompose.
    Returns:
        L ((n,n) ndarray): The lower-triangular part of the decomposition.
        U ((n,n) ndarray): The upper-triangular part of the decomposition.
    """
    U = np.copy(A) #copy A
    U = U.astype(np.float64)
    rows, cols = A.shape[0], A.shape[1] #get dimensions
    L = np.eye(rows) #create appropriate identity
    for col in range(0, cols):
        for row in range (col + 1, rows):
            if U[col,col] == 0:
                continue
            L[row][col] = U[row, col] / U[col, col] #compute element of L
            U[row][col:] = U[row][col:] - (L[row][col] * U[col][col:]) #compute row of U
    return L, U

# Problem 3
def solve(A, b):
    """This function uses the LU decomposition and back substitution to solve Ax = b
    Parameters:
        A ((n,n) ndarray)
        b ((n,) ndarray)
    Returns:
        x ((m,) ndarray): The solution to the linear system.
    """
    L, U = lu(A) #get LU decomp of A
    rows, cols = A.shape[0], A.shape[1]
    y = np.zeros(rows)
    x = np.zeros(rows)

    #compute y
    for row in range(0, rows):
        y[row] = b[row] #intialize y_j as b_j
        for col in range(0, row): # subtract off the appropriate scalar multiple of previous elements of y
            y[row] -= L[row][col] * y[col]
    #compute x
    #we start from the top and work down
    for row in range(rows-1, -1, -1):
        x[row] = y[row] #set elements
        for col in range(row+1, rows): #modify the new element with previously calculated elements
            x[row] -= U[row][col] * x[col]
        x[row] *= 1/U[row][row]

    return x

# Problem 4
def prob4():
    """Time different scipy.linalg functions for solving square linear systems.
    """
    sizes = 2 ** np.arange(1,8)
    time_la_inv = []
    time_la_solve = []
    time_lu_factor = []
    time_lu_solve = []

    for n in sizes:
        b = np.random.random((n,1))
        A = np.random.random((n,n))

        #timing inverse
        start = time.time()
        x = la.inv(A) @ b
        end = time.time()
        time_la_inv.append(end - start)

        #timing la solve
        start = time.time()
        x = la.solve(A, b)
        end = time.time()
        time_la_solve.append(end - start)

        #timing lu solve (with factorization)
        start = time.time()
        L, P = la.lu_factor(A)
        x = la.lu_solve((L,P), b)
        end = time.time()
        time_lu_factor.append(end - start)

        #timing lu solve (without factorization)
        L, P = la.lu_factor(A)
        start = time.time()
        x = la.lu_solve((L,P), b)
        end = time.time()
        time_lu_solve.append(end - start)

    #plot times using log scale
    fig = plt.figure()
    fig.set_dpi(2000)
    ax = fig.add_subplot(111)
    ax.loglog(sizes, time_la_inv, label = 'Inverse')
    ax.loglog(sizes, time_la_solve, label = 'Solve')
    ax.loglog(sizes, time_lu_factor, label = 'LU Factorization')
    ax.loglog(sizes, time_lu_solve, label = 'LU Solve')
    ax.set_yscale('log', base=2)
    ax.set_xscale('log', base=2)
    plt.title('Time Comparisons For Solution Methods')
    plt.xlabel('Matrix - Vector Size')
    plt.ylabel('Time (s)')
    plt.legend(loc = 'best')
    plt.show()

# Problem 5
def prob5(n):
    """
    Parameters:
        n (int): Dimensions of the sparse matrix B.

    Returns:
        A ((n**2,n**2) SciPy sparse matrix)
    """
    #set up a sparse little matrix
    B = sparse.lil_matrix((n, n))
    #set the diagnonal elements to 4
    B.setdiag(-4)
    #set up the 1s and -1s next to the diagonal
    B.setdiag(1, -1)
    B.setdiag(1, 1)
    #set up a sparse block diagonal matrix with B
    A = sparse.block_diag([B] * n)
    #set up identity matrix around B
    A.setdiag([1] * n**2, n)
    A.setdiag([1] * n**2, -n)
    return A

# Problem 6
def prob6():
    """Times regular and sparse linear system solvers.
    """
    #initializing empty arrays and domain of sizes
    domain = 2 ** np.arange(1,6)
    CSR_time = []
    array_time = []

    for n in domain:
        A = prob5(n)
        b = np.random.random(n**2)
        Acsr = A.tocsr()

        #time sparse solver
        start = time.time()
        x_1 = spla.spsolve(Acsr, b)
        end = time.time()
        CSR_time.append(end - start)

        #time normal array solver
        start = time.time()
        x_2 = la.solve(A.toarray(), b)
        end = time.time()
        array_time.append(end - start)

    #plot times using log scale
    fig = plt.figure()
    fig.set_dpi(150)
    ax = fig.add_subplot(111)
    ax.loglog(domain, CSR_time, label = 'Sparse solver')
    ax.loglog(domain, array_time, label = 'Array solver')
    ax.set_title('Sparse vs Array solver times')
    ax.set_xlabel('Sizes')
    ax.set_ylabel('Time(s)')
    ax.legend(loc='best')
    plt.show()
