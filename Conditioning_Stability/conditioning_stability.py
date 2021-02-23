    # condition_stability.py
"""Volume 1: Conditioning and Stability.
Caelan osman
Math 347 Sec. 2
Feb. 7, 2021
"""

import numpy as np
import sympy as sy
from scipy import linalg as la
from matplotlib import pyplot as plt


# Problem 1
def matrix_cond(A):
    """Calculate the condition number of A with respect to the 2-norm."""

    #get singular values
    singular_vals = la.svdvals(A)

    minimum = np.min(singular_vals)
    if minimum == 0:
        return np.inf
    else:
        return np.max(singular_vals) / minimum


# Problem 2
def prob2():
    """Randomly perturb the coefficients of the Wilkinson polynomial by
    replacing each coefficient c_i with c_i*r_i, where r_i is drawn from a
    normal distribution centered at 1 with standard deviation 1e-10.
    Plot the roots of 100 such experiments in a single figure, along with the
    roots of the unperturbed polynomial w(x).

    Returns:
        (float) The average absolute condition number.
        (float) The average relative condition number.
    """
    w_roots = np.arange(1, 21)

    # Get the exact Wilkinson polynomial coefficients using SymPy.
    x, i = sy.symbols('x i')
    w = sy.poly_from_expr(sy.product(x-i, (i, 1, 20)))[0]
    w_coeffs = np.array(w.all_coeffs())


    absolute_condition = np.zeros(100)
    relative_condition = np.zeros(100)
    #repeat 100 times
    n = w_coeffs.size
    plt.plot(w_roots, np.zeros(n - 1), 'bo', markersize=5, label='Original')
    for i in range(100):
        #perturbation coefficients
        h = np.random.normal(loc=1, scale=1e-10, size=n)
        #perturb the coeffecients
        new_coeffs = w_coeffs * h
        #get the new perturbed roots
        new_roots = np.sort(np.roots(np.poly1d(new_coeffs)))

        #plot
        if i == 99:
            plt.plot(np.real(new_roots), np.imag(new_roots), 'k.', markersize=2, label='Perturbed')
        else:
            plt.plot(np.real(new_roots), np.imag(new_roots), 'k.', markersize=2)

        #get condition numbers
        absolute_condition[i] = la.norm(new_roots - w_roots, np.inf) / la.norm(h, np.inf)
        relative_condition[i] = absolute_condition[i] * la.norm(w_coeffs, np.inf) / la.norm(w_roots, np.inf)


    plt.legend(loc='best')
    plt.xlabel('Real Axis')
    plt.ylabel('Imaginary Axis')
    plt.show()

    #calculate and return averages
    return np.mean(absolute_condition), np.mean(relative_condition)




# Problem 3
def eig_cond(A):
    """Approximate the condition numbers of the eigenvalue problem at A.

    Parameters:
        A ((n,n) ndarray): A square matrix.

    Returns:
        (float) The absolute condition number of the eigenvalue problem at A.
        (float) The relative condition number of the eigenvalue problem at A.
    """
    #compute perturbation matrix
    reals = np.random.normal(0, 1e-10, A.shape)
    imags = np.random.normal(0, 1e-10, A.shape)
    H = reals + 1j*imags

    #comptue eigenvalues of A and A + H

    eigvals_A = la.eigvals(A)
    eigvals_AH = la.eigvals(A + H)

    #compute and retur absolute and relative condition numbers
    absolute = la.norm(eigvals_A - eigvals_AH, ord=2) / la.norm(H, ord=2)
    relative = la.norm(A, ord=2) * absolute / la.norm(eigvals_A, ord=2)

    return absolute, relative


# Problem 4
def prob4(domain=[-100, 100, -100, 100], res=50):
    """Create a grid [x_min, x_max] x [y_min, y_max] with the given resolution. For each
    entry (x,y) in the grid, find the relative condition number of the
    eigenvalue problem, using the matrix   [[1, x], [y, 1]]  as the input.
    Use plt.pcolormesh() to plot the condition number over the entire grid.

    Parameters:
        domain ([x_min, x_max, y_min, y_max]):
        res (int): number of points along each edge of the grid.
    """
    #create mesh
    x_vals = np.linspace(domain[0], domain[1], res)
    y_vals = np.linspace(domain[2], domain[3], res)
    X_mesh, Y_mesh = np.meshgrid(x_vals, y_vals)

    #get condition numbers
    condition_vals = np.zeros(X_mesh.shape)
    for i, col in enumerate(X_mesh):
        for j, element in enumerate(col):
            A = np.array([[1, element],
                          [Y_mesh[i, j], 1]])

            condition_vals[i, j] = eig_cond(A)[1]

    #plot
    plt.pcolormesh(X_mesh, Y_mesh, condition_vals, cmap='gray_r')
    plt.colorbar()
    plt.show()


# Problem 5
def prob5(n):
    """Approximate the data from "stability_data.npy" on the interval [0,1]
    with a least squares polynomial of degree n. Solve the least squares
    problem using the normal equation and the QR decomposition, then compare
    the two solutions by plotting them together with the data. Return
    the mean squared error of both solutions, ||Ax-b||_2.

    Parameters:
        n (int): The degree of the polynomial to be used in the approximation.

    Returns:
        (float): The forward error using the normal equations.
        (float): The forward error using the QR decomposition.
    """
    xk, yk = np.load("stability_data.npy").T
    #get vandermonde matrix
    A = np.vander(xk, n+1)
    #solve vandermonde matrix normal equations system
    xv = la.inv(A.T @ A) @ A.T @ yk

    #solve system using qr decomposition
    Q, R = la.qr(A, mode='economic')
    xt = la.solve_triangular(R, Q.T @ yk)

    #plot
    domain = np.linspace(0, 1, 200)
    #get polynomials
    y1 = np.polyval(xv, domain)
    y2 = np.polyval(xt, domain)
    #plot values
    plt.plot(domain, y1, label='Vandermonde')
    plt.plot(domain, y2, label='QR')
    plt.plot(xk, yk, 'o', label='Data')
    plt.ylim([0, 4])
    plt.legend(loc='best')
    plt.show()

    #compute and return forward error
    return la.norm(A @ xv - yk, ord=2), la.norm(A @ xt - yk, ord=2)


# Problem 6
def prob6():
    """For n = 5, 10, ..., 50, compute the integral I(n) using SymPy (the
    true values) and the subfactorial formula (may or may not be correct).
    Plot the relative forward error of the subfactorial formula for each
    value of n. Use a log scale for the y-axis.
    """

    #helper function
    def rel_forward_error(n):
        #create symbols
        x = sy.symbols('x')
        #create current expression
        expression = np.e**(x-1)*x**n
        #get integration value
        integral_val = float(sy.integrate(expression, (x, 0, 1)))
        #compute the value directly
        direct_val = float((-1)**n * (sy.subfactorial(n) - (sy.factorial(n) / np.e)))
        #calculate and return relative forward error
        return np.abs(integral_val - direct_val) / np.abs(integral_val)


    #domain for plotting
    domain = np.arange(5, 55, 5)
    #create array of relative forward error
    rel_error = np.array([rel_forward_error(n) for n in domain])

    #plot
    plt.semilogy(domain, rel_error, 'g:')
    plt.xlabel('n values')
    plt.ylabel('relative error')
    plt.show()




if __name__ == "__main__":

    #problem 1:
    """
    #test 1: singular matrix
    S = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
    condition = matrix_cond(S)
    print(condition)
    print(np.linalg.cond(S))
    #test 2: orthogonal matrix
    A = np.random.random((3, 3))
    Q, R = la.qr(A)
    condition = matrix_cond(Q)
    print(condition)
    print(np.linalg.cond(Q))
    """

    #problem 2:
    #print(prob2())

    #test for prob3 and prob4
    #prob4(res=200)

    #prob5
    #print(prob5(15))

    #prob6
    #prob6()
