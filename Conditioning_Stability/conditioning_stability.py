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
        h = np.random.normal(loc=1, scale=10e-10, size=n)
        #perturb the coeffecients
        new_coeffs = w_coeffs * h
        #get the new perturbed roots
        new_roots = np.roots(np.poly1d(new_coeffs))

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
    raise NotImplementedError("Problem 3 Incomplete")


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
    raise NotImplementedError("Problem 4 Incomplete")


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
    raise NotImplementedError("Problem 5 Incomplete")


# Problem 6
def prob6():
    """For n = 5, 10, ..., 50, compute the integral I(n) using SymPy (the
    true values) and the subfactorial formula (may or may not be correct).
    Plot the relative forward error of the subfactorial formula for each
    value of n. Use a log scale for the y-axis.
    """
    raise NotImplementedError("Problem 6 Incomplete")


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

