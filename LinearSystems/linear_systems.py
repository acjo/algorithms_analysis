# linear_systems.py
"""Volume 1: Linear Systems.
Caelan Osman
Math 345 Sec 3
October 19, 2020
"""

import numpy as np

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

    For various values of n, generate a random nxn matrix A and a random
    n-vector b using np.random.random(). Time how long it takes to solve the
    system Ax = b with each of the following approaches:

        1. Invert A with la.inv() and left-multiply the inverse to b.
        2. Use la.solve().
        3. Use la.lu_factor() and la.lu_solve() to solve the system with the
            LU decomposition.
        4. Use la.lu_factor() and la.lu_solve(), but only time la.lu_solve()
            (not the time it takes to do the factorization).

    Plot the system size n versus the execution times. Use log scales if
    needed.
    """
    raise NotImplementedError("Problem 4 Incomplete")


# Problem 5
def prob5(n):
    """Let I be the n × n identity matrix, and define
                    [B I        ]        [-4  1            ]
                    [I B I      ]        [ 1 -4  1         ]
                A = [  I . .    ]    B = [    1  .  .      ],
                    [      . . I]        [          .  .  1]
                    [        I B]        [             1 -4]
    where A is (n**2,n**2) and each block B is (n,n).
    Construct and returns A as a sparse matrix.

    Parameters:
        n (int): Dimensions of the sparse matrix B.

    Returns:
        A ((n**2,n**2) SciPy sparse matrix)
    """
    raise NotImplementedError("Problem 5 Incomplete")


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
