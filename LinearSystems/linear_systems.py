# linear_systems.py
"""Volume 1: Linear Systems.
Caelan Osman
Math 345 Sec 3
October 19, 2020
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg as la
from scipy import sparse
import time
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
    plt.loglog(sizes, time_la_inv, label = 'Inverse', basex = 2, basey = 2)
    plt.loglog(sizes, time_la_solve, label = 'Solve', basex = 2, basey = 2)
    plt.loglog(sizes, time_lu_factor, label = 'LU Factorization', basex = 2, basey = 2)
    plt.loglog(sizes, time_lu_solve, label = 'LU Solve', basex = 2, basey = 2)
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
    #using to create B and I matrix
    main_1 = [1 for i in range(0, n-1)]
    main = [-4 for i in range (0, n)]
    diagonals = [main_1, main, main_1]
    offsets = [-1, 0, 1]
    main_2 = [1 for i in range (0, n)]


    if n > 1:
        #populate B and I matrix
        B = sparse.diags(diagonals, offsets, shape = (n,n))
        I = sparse.diags([main_2], [0], shape = (n,n))
        final_diag = []

        #creates 2d list as an input for sparse bmat
        for i in range (0, n):
            current = []
            for j in range(0, n):
                print(i,j)
                if i == 0:
                    if j == 0: #B will be at the beginning with None populating the rest
                        current.append(B)
                    else:
                        current.append(None)
                elif i == n - 1: #B will be at the end with None populating the rest
                    if j == n - 1:
                        current.append(B)
                    else:
                        current.append(None)
                else:#B will be in the diagonal with I to the left and right of it, None everywhere else
                    if j == i - 1:
                        current.append(I)
                    elif j == i + 1:
                        current.append(I)
                    elif j == i:
                        current.append(B)
                    else:
                        current.append(None)
            final_diag.append(current)

        A = sparse.bmat(final_diag, format='bsr') #create and return the spares matrix
        return A
    elif n == 1: #if size is 1 just return the B matrix
        B = sparse.diags(diagonals, offsets, shape = (n,n))
        final_diag = [B]
        A = sparse.block_diag((tuple(final_diag)))
        return A
    else: #otherwise return an empty matrix
        A = np.array([])
        return A


# Problem 6
def prob6():
    """Time regular and sparse linear system solvers.

    For various values of n, generate the (n**2,n**2) matrix A described of
    prob5() and vector b of length n**2. Time how long it takes to solve the
    system Ax = b with each of the following approaches:

        1. Convert A to CSR format and use scipy.sparse.linalg.spsolve()
        2. Convert A to a NumPy array and use scipy.linalg.solve().

    In each experiment, only time how long it takes to solve the system (not
    how long it takes to convert A to the appropriate format). Plot the system
    size n**2 versus the execution times. As always, use log scales where
    appropriate and use a legend to label each line.
    """
    raise NotImplementedError("Problem 6 Incomplete")
