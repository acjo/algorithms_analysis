# lstsq_eigs.py
"""Volume 1: Least Squares and Computing Eigenvalues.
Caelan Osman
Math 345 Sec 3
October 26, 2020
"""

# (Optional) Import functions from your QR Decomposition lab.
# import sys
# sys.path.insert(1, "../QR_Decomposition")
# from qr_decomposition import qr_gram_schmidt, qr_householder, hessenberg

import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg as la


# Problem 1
def least_squares(A, b):
    """Calculate the least squares solutions to Ax = b by using the QR
    decomposition.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n <= m.
        b ((m, ) ndarray): A vector of length m.

    Returns:
        x ((n, ) ndarray): The solution to the normal equations.
    """
    Q, R = la.qr(A, mode='economic') #get QR decomp of A
    temp = Q.T @ b
    sol = la.solve_triangular(R, temp, lower=False) #solve least squares problem
    return sol

# Problem 2
def line_fit():
    """Find the least squares line that relates the year to the housing price
    index for the data in housing.npy. Plot both the data points and the least
    squares line.
    """
    Data = np.load('housing.npy') #load data
    years = Data[:, 0] #get x axis data
    prices = Data[:, 1] #get y axis data

    #get the least squares solution
    #this makes our A matrix will be a n x 2
    A = np.concatenate((years.reshape((len(years), 1)), np.ones(len(years)).reshape(len(years), 1)), axis = 1)
    a, b = least_squares(A, prices) #this finds the least squars solution

    #defines the function we will use to plot
    line = lambda x: a * x + b

    plt.plot(years, prices, '*', label='Discrete Data') #plot the scatter plot, discrete data
    plt.plot(years, line(years), '-', label='Line of Best Fit') #plots our best fit line
    plt.legend(loc='best')
    plt.xlabel('Year (+2000)')
    plt.ylabel('Prices')
    plt.title('Housing market')
    plt.show()


# Problem 3
def polynomial_fit():
    """Find the least squares polynomials of degree 3, 6, 9, and 12 that relate
    the year to the housing price index for the data in housing.npy. Plot both
    the data points and the least squares polynomials in individual subplots.
    """
    #grabbing data from the file
    data = np.load('housing.npy')
    years = data[:, 0]
    prices = data[:, 1]

    #creating vandermonde matrices
    A_3 = np.vander(years, 4)
    A_6 = np.vander(years, 7)
    A_9 = np.vander(years, 10)
    A_12 = np.vander(years, 13)

    #solving least sqaures problems
    sol_3, _, _, _ = la.lstsq(A_3, prices)
    sol_6, _, _, _ = la.lstsq(A_6, prices)
    sol_9, _, _, _ = la.lstsq(A_9, prices)
    sol_12, _, _, _ = la.lstsq(A_12, prices)

    #setting least squares poly functions
    p_3 = np.poly1d(sol_3)
    p_6 = np.poly1d(sol_6)
    p_9 = np.poly1d(sol_9)
    p_12 = np.poly1d(sol_12)

    #setting domain to plot polynomials
    domain = np.linspace(0, 16, 150)

    ax1 = plt.subplot(231)
    ax1.plot(years, prices, '*', label='Discrete Data') #plot the scatter plot, discrete data
    ax1.set_title('Discrete')
    plt.xlabel('Year (+2000)')
    plt.ylabel('Prices')

    #degree 3
    ax2 = plt.subplot(232)
    ax2.plot(domain, p_3(domain), label = 'P3')
    ax2.set_title('Degree 3 polynomial')
    plt.xlabel('Year (+2000)')
    plt.ylabel('Prices')

    #degree 6
    ax3 = plt.subplot(233)
    ax3.plot(domain, p_6(domain), label = 'P6')
    ax3.set_title('Degree 6 polynomial')
    plt.xlabel('Year (+2000)')
    plt.ylabel('Prices')

    #degree 9
    ax4 = plt.subplot(234)
    ax4.plot(domain, p_9(domain), label = 'P9')
    ax4.set_title('Degree 9 polynomial')
    plt.xlabel('Year (+2000)')
    plt.ylabel('Prices')

    #degree 12
    ax5 = plt.subplot(235)
    ax5.plot(domain, p_12(domain), label = 'P12')
    ax5.set_title('Degree 12 polynomial')
    plt.xlabel('Year (+2000)')
    plt.ylabel('Prices')

    '''
    comparing to polyfit

    c_12 = np.polyfit(years, prices, 12)
    poly_12 = lambda x: c_12[12] * x**12 + c_12[11] * x**11+ c_12[10] * x**10+ c_12[9] * x**9+ c_12[8] * x**8+ c_12[7] * x**7+ c_12[6] * x**6+ c_12[5] * x**5+ c_12[4] * x**4+ c_12[3] * x**3+ c_12[2] * x**2+ c_12[1] * x+ c_12[0]

    ax6 = plt.subplot(236)
    ax6.plot(domain, poly_12(domain), label = 'poly 12')
    ax5.set_title('Degree 12 polynomial polyfit')
    plt.xlabel('Year (+2000)')
    plt.ylabel('Prices')
    '''

    plt.legend(loc='best')
    plt.suptitle('Housing market')
    plt.show()


def plot_ellipse(a, b, c, d, e):
    """Plot an ellipse of the form ax^2 + bx + cxy + dy + ey^2 = 1."""
    theta = np.linspace(0, 2*np.pi, 200)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    A = a*(cos_t**2) + c*cos_t*sin_t + e*(sin_t**2)
    B = b*cos_t + d*sin_t
    r = (-B + np.sqrt(B**2 + 4*A)) / (2*A)

    plt.plot(r*cos_t, r*sin_t)
    plt.gca().set_aspect("equal", "datalim")

# Problem 4
def ellipse_fit():
    """Calculate the parameters for the ellipse that best fits the data in
    ellipse.npy. Plot the original data points and the ellipse together, using
    plot_ellipse() to plot the ellipse.
    """
    raise NotImplementedError("Problem 4 Incomplete")


# Problem 5
def power_method(A, N=20, tol=1e-12):
    """Compute the dominant eigenvalue of A and a corresponding eigenvector
    via the power method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The maximum number of iterations.
        tol (float): The stopping tolerance.

    Returns:
        (float): The dominant eigenvalue of A.
        ((n,) ndarray): An eigenvector corresponding to the dominant
            eigenvalue of A.
    """
    raise NotImplementedError("Problem 5 Incomplete")


# Problem 6
def qr_algorithm(A, N=50, tol=1e-12):
    """Compute the eigenvalues of A via the QR algorithm.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The number of iterations to run the QR algorithm.
        tol (float): The threshold value for determining if a diagonal S_i
            block is 1x1 or 2x2.

    Returns:
        ((n,) ndarray): The eigenvalues of A.
    """
    raise NotImplementedError("Problem 6 Incomplete")
