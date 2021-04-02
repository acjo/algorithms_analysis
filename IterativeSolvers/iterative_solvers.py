# iterative_solvers.py
"""Volume 1: Iterative Solvers.
Caelan Osman
Math 347 Sec. 2
March 30, 2021
"""

import numpy as np
from scipy import sparse
from scipy import linalg as la
from matplotlib import pyplot as plt
from scipy.sparse import linalg as spla

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

    #x0 = np.ones(D_inv.size)
    x0 = np.random.random(D_inv.size)

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
            #update previous
            x0 = x1

        #plot absolute convergence rates on logplot
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
            #update previous
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
    #x0 = np.ones(n)
    x0 = np.random.random(n)

    #if we want to plot
    if plot:
        #intialize error vector
        error = []
        error.append(la.norm(A@x0 - b, ord=np.inf))
        for i in range(maxiter):
            #update x1
            x1 = np.array([x0[j] + (b[j] - np.inner(A[j, :], x0)) /A[j, j] for j in range(n)])
            error.append(la.norm(A@x1 - b, ord=np.inf))
            #check convergence
            if la.norm(x0-x1, ord=np.inf) < tol:
                break
            #update previous
            x0 = x1

        #plot absolute convergence rates on logplot
        plt.semilogy(np.arange(0, i+2), error, 'm-')
        plt.title('Convergence of Gauss Seidel Method')
        plt.xlabel('Iteration Count')
        plt.ylabel('Absolute Error of Approximation')
        plt.show()

    #if we don't want to plot
    else:
        for i in range(maxiter):
            #update x1
            return A[0, :], x0
            x1 = np.array([x0[j] + (b[j] - np.inner(A[j, :], x0)) /A[j, j] for j in range(n)])
            #check convergence
            if la.norm(x0-x1, ord=np.inf) < tol:
                break
            #update previous
            x0 = x1
            break

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
    #initialze the vectors
    n = b.size
    x0 = np.random.random(n)
    #x0 = np.ones(n)
    #compute iterations
    for i in range(maxiter):
        #iterate x1
        x1 = np.array([x0[j] +
                       (b[j] - np.inner(A.data[A.indptr[j]:A.indptr[j+1]],
                                        x0[A.indices[A.indptr[j]:A.indptr[j+1]]]))
                       / A[j, j] for j in range(n)])
        #check convergence
        if la.norm(x0-x1, ord=np.inf) < tol:
            break

        #update previous
        x0 = x1

    return x1

# Problem 5
def sor(A, b, omega, tol=1e-8, maxiter=100):
    """Calculate the solution to the system Ax = b via Successive Over-
    Relaxation.

    Parameters: A ((n, n) csr_matrix): A (n, n) sparse matrix.
        b ((n, ) Numpy Array): A vector of length n.
        omega (float in [0,1]): The relaxation factor.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.

    Returns:
        ((n,) ndarray): The solution to system Ax = b.
        (bool): Whether or not Newton's method converged.
        (int): The number of iterations computed.
    """
    #TODO: FInd out if omega is between 0 and 1 or greater than or equal to 1
    #initialze the vectors and boolean
    converged = False
    n = b.size
    x0 = np.random.random(n)
    #x0 = np.ones(n)
    #compute iterations
    for i in range(maxiter):
        #iterate x1
        x1 = np.array([x0[j] +
                       (b[j] - np.inner(A.data[A.indptr[j]:A.indptr[j+1]],
                                        x0[A.indices[A.indptr[j]:A.indptr[j+1]]]))
                       * omega / A[j, j] for j in range(n)])
        #check convergence
        if la.norm(x0-x1, ord=np.inf) < tol:
            #update convergance boolean
            converged = True
            break

        #update previous
        x0 = x1

    return x1, converged, i+1


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
    #subroutine to create the A matrix
    def create_A(m):
        #set up a sparse little matrix
        B = sparse.lil_matrix((m, m))
        #set the diagnonal elements to 4
        B.setdiag(-4)
        #set up the 1s and -1s next to the diagonal
        B.setdiag(1, -1)
        B.setdiag(1, 1)
        #set up a sparse block diagonal matrix with B
        A = sparse.block_diag([B] * m)
        #set up identity matrix around B
        A.setdiag([1] * m**2, m)
        A.setdiag([1] * m**2, -m)
        return A.tocsr()

    #subroutine to create the u vector
    def create_b(m):
        #notice this pattern repeats
        a = np.zeros(m)
        a[0] = -100
        a[-1] = -100

        return np.tile(a, m)

    A = create_A(n)
    b = create_b(n)

    u, convergence, iters = sor(A, b, omega, tol=tol, maxiter=maxiter)


    if plot:
        U = u.reshape((n, n))
        #FIXME: Ask about creating grid
        upper = np.max(u)
        lower = np.min(u)
        x = np.linspace(0, 10, n)
        y = x.copy()
        X, Y = np.meshgrid(x, y)

        plt.pcolormesh(X, Y, U, cmap='coolwarm')
        plt.show()


    return u, convergence, iters



# Problem 7
def prob7():
    """Run hot_plate() with omega = 1, 1.05, 1.1, ..., 1.9, 1.95, tol=1e-2,
    and maxiter = 1000 with A and b generated with n=20. Plot the iterations
    computed as a function of omega.
    """
    #FIXME: Omega does some weird stuff
    omega = np.linspace(1, 1.95, 20)
    iteration_count = []
    converged = []

    for w in omega:
        _, c, i = hot_plate(20, w, tol=1e-2, maxiter=100, plot=False)
        iteration_count.append(i)
        converged.append(c)


    plt.plot(omega, iteration_count, 'r:')
    plt.title(r'SOR Iterations vs. $\omega$')
    plt.xlabel(r'$\omega$')
    plt.ylabel('Iteration Number')
    plt.show()

    min_index = np.argmin(iteration_count)

    return omega[min_index]




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

    #prob4
    '''
    b = np.random.random(15000)
    A = diag_dom(15000)
    A = sparse.csr_matrix(A)

    x4 = gauss_seidel_sparse(A, b, tol=1e-12, maxiter=150)
    print(iters)
    print(np.allclose(A @ x4, b))
    '''

    #prob5 can be tested like prob4

    #prob6
    '''
    u, c, i = hot_plate(50, 0.5, tol=1e-8, maxiter=1000, plot=True)
    print("converged:", c)
    print("iterations:", i)
    '''

    #prob7
    print(prob7())
