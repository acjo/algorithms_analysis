# iterative_solvers.py
"""Volume 1: Iterative Solvers.
Caelan Osman
Math 347 Sec. 2
March 30, 2021
"""

import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt

# Helper function
def diag_dom(n, num_entries=None):
    """Generate a strictly diagonally dominant (n, n) matrix.

    Parameters:
        n (int): The dimension of the system.
        num_entries (int): The number of nonzero values.
            Defaults to n^(3/2)-n.

    Returns:
        A ((n,n) ndarray): A (n, n) strictly diagonally dominant matrix.
    """
    if num_entries is None:
        num_entries = int(n**1.5) - n
    A = np.zeros((n,n))
    rows = np.random.choice(np.arange(0,n), size=num_entries)
    cols = np.random.choice(np.arange(0,n), size=num_entries)
    data = np.random.randint(-4, 4, size=num_entries)
    for i in range(num_entries):
        A[rows[i], cols[i]] = data[i]
    for i in range(n):
        A[i,i] = np.sum(np.abs(A[i])) + 1
    return A

# Problems 1 and 2
def jacobi(A, b, tol=1e-8, maxiter=100, plot=False):
    """Calculate the solution to the system Ax = b via the Jacobi Method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        b ((n ,) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.

    Returns:
        ((n,) ndarray): The solution to system Ax = b.
    """
    #get diagonal, Lower and upper triangular matrices
    D_inv = 1/np.diag(A)

    x0 = np.ones(D_inv.size)
    #TODO ask about initial point
    #TODO ask about which convergence check to use

    #if we want to plot aboslute error
    if plot:
        #error vector
        error = []
        error.append(la.norm(A@x0 - b, ord=np.inf))
        for i in range(maxiter):
            #create next iteration
            x1 = x0 + D_inv * (b - A @ x0)
            #append error
            error.append(la.norm(A@x1 - b, ord=np.inf))
            #check convergence
            if la.norm(x0 - x1, ord=np.inf) < tol:
                break
            #update previous iteration
            x0 = x1

        #plot on log plot absolute convergence rates
        plt.semilogy(np.arange(0, i+2), error, 'm-')
        plt.title('Convergence of Jacobi Method')
        plt.xlabel('Iteration Count')
        plt.ylabel('Absolute Error of Approximation')
        plt.show()

    #if we don't want to plot aboslute error
    else:
        for i in range(maxiter):
            #create next iteration
            x1 = x0 + D_inv * (b - A @ x0)
            #check convergence
            if la.norm(x0 - x1, ord=np.inf) < tol:
                break
            #update previous iteration
            x0 = x1

    return x1

# Problem 3
def gauss_seidel(A, b, tol=1e-8, maxiter=100, plot=False):
    """Calculate the solution to the system Ax = b via the Gauss-Seidel Method.

    Parameters:
        A ((n, n) ndarray): A square matrix.
        b ((n, ) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.
        plot (bool): If true, plot the convergence rate of the algorithm.

    Returns:
        x ((n,) ndarray): The solution to system Ax = b.
    """
    #initialize vectors
    n = A.shape[1]
    x0 = np.ones(n)

    #if we want to plot
    if plot:
        #intialize error vector
        error = []
        error.append(la.norm(A@x0 - b, ord=np.inf))
        for i in range(maxiter):
            #update x0
            x1 = np.array([x0[j] + (b[j] - np.inner(A[j, :], x0)) /A[j, j] for j in range(n)])
            error.append(la.norm(A@x1 - b, ord=np.inf))
            #check convergence
            if la.norm(x0-x1, ord=np.inf) < tol:
                break
            x0 = x1

        plt.semilogy(np.arange(0, i+2), error, 'm-')
        plt.title('Convergence of Gauss Seidel Method')
        plt.xlabel('Iteration Count')
        plt.ylabel('Absolute Error of Approximation')
        plt.show()

    #if we don't want to plot
    else:
        for i in range(maxiter):
            #update x0
            x1 = np.array([x0[j] + (b[j] - np.inner(A[j, :], x0)) /A[j, j] for j in range(n)])
            #check convergence
            if la.norm(x0-x1, ord=np.inf) < tol:
                break
            x0 = x1

    return x1


# Problem 4
def gauss_seidel_sparse(A, b, tol=1e-8, maxiter=100):
    """Calculate the solution to the sparse system Ax = b via the Gauss-Seidel
    Method.

    Parameters:
        A ((n, n) csr_matrix): A (n, n) sparse CSR matrix.
        b ((n, ) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): the maximum number of iterations to perform.

    Returns:
        x ((n,) ndarray): The solution to system Ax = b.
    """
    raise NotImplementedError("Problem 4 Incomplete")


# Problem 5
def sor(A, b, omega, tol=1e-8, maxiter=100):
    """Calculate the solution to the system Ax = b via Successive Over-
    Relaxation.

    Parameters:
        A ((n, n) csr_matrix): A (n, n) sparse matrix.
        b ((n, ) Numpy Array): A vector of length n.
        omega (float in [0,1]): The relaxation factor.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.

    Returns:
        ((n,) ndarray): The solution to system Ax = b.
        (bool): Whether or not Newton's method converged.
        (int): The number of iterations computed.
    """
    raise NotImplementedError("Problem 5 Incomplete")


# Problem 6
def hot_plate(n, omega, tol=1e-8, maxiter=100, plot=False):
    """Generate the system Au = b and then solve it using sor().
    If show is True, visualize the solution with a heatmap.

    Parameters:
        n (int): Determines the size of A and b.
            A is (n^2, n^2) and b is one-dimensional with n^2 entries.
        omega (float in [0,1]): The relaxation factor.
        tol (float): The iteration tolerance.
        maxiter (int): The maximum number of iterations.
        plot (bool): Whether or not to visualize the solution.

    Returns:
        ((n^2,) ndarray): The 1-D solution vector u of the system Au = b.
        (bool): Whether or not Newton's method converged.
        (int): The number of computed iterations in SOR.
    """
    raise NotImplementedError("Problem 6 Incomplete")


# Problem 7
def prob7():
    """Run hot_plate() with omega = 1, 1.05, 1.1, ..., 1.9, 1.95, tol=1e-2,
    and maxiter = 1000 with A and b generated with n=20. Plot the iterations
    computed as a function of omega.
    """
    raise NotImplementedError("Problem 7 Incomplete")



if __name__ == "__main__":

    #prob1 and 2
    '''
    b = np.random.random(100)
    A = diag_dom(100)
    x = jacobi(A, b)
    print(np.allclose(A@x, b))
    x1 = jacobi(A, b, plot=True)
    print(np.allclose(A@x1, b))
    print(np.allclose(x, x1))
    '''

    #prob3
    '''
    b = np.random.random(30)
    A = diag_dom(30)
    x2 = gauss_seidel(A, b, plot=False)
    print(np.allclose(A@x2, b))
    x3 = gauss_seidel(A, b, plot=True)
    print(np.allclose(A@x3, b))
    print(np.allclose(x2, x3))
    '''
