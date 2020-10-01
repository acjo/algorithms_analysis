# linear_transformations.py
"""Volume 1: Linear Transformations.
Caelan Osman
Math 345 Sec 3
September 29, 2020
"""

from random import random
import numpy as np
from matplotlib import pyplot as plt


# Problem 1
def stretch(A, a, b):
    """This function scales the points in A by a in the x direction and b in the
    y direction.
    """
    return np.array([[a,0], [0, b]]) @ A


def shear(A, a, b):
    """This function slants the points in A by a in the x direction and b in the
    y direction.
    """
    return np.array([[1,a], [b,1]]) @ A

def reflect(A, a, b):
    """This function reflects the points in A about the line that passes through the origin
    and the point (a,b).
    """
    return 1/(a**2 + b**2) * (np.array([[a**2 - b**2, 2 * a * b], [2 * a * b, b**2 - a**2]]) @ A)

def rotate(A, theta):
    """This function rotates the points in A about the origin by theta radians.
    """
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) @ A

def plot_comparison():
    data = np.load('horse.npy')

    #figure
    ax1 = plt.subplot(231)
    ax1.plot(data[0], data[1], 'k,')
    plt.axis([-1,1,-1,1])
    plt.title("Original", fontsize = 10)
    plt.gca().set_aspect("equal")

    ax2 = plt.subplot(232)
    ax2.plot(stretch(data, 1/2, 6/5)[0], stretch(data, 1/2, 6/5)[1], 'k,')
    plt.axis([-1,1,-1,1])
    plt.title("Stretch", fontsize = 10)
    plt.gca().set_aspect("equal")


    ax3 = plt.subplot(233)
    ax3.plot(shear(data, 1/2, 0)[0], shear(data, 1/2, 0)[1], 'k,')
    plt.axis([-1,1,-1,1])
    plt.title("Shear", fontsize = 10)
    plt.gca().set_aspect("equal")

    ax4 = plt.subplot(234)
    ax4.plot(reflect(data, 0, 1)[0], reflect(data, 0,1)[1], 'k,')
    plt.axis([-1,1,-1,1])
    plt.title("Reflection", fontsize = 10)
    plt.gca().set_aspect("equal")

    ax5 = plt.subplot(235)
    ax5.plot(rotate(data, np.pi/2)[0], rotate(data, np.pi/2)[1], 'k,')
    plt.axis([-1,1,-1,1])
    plt.title("Rotation", fontsize = 10)
    plt.gca().set_aspect("equal")

    '''Composition:
    ax6 = plt.subplot(236)
    ax6.plot(new[0], new[1], 'k,')
    plt.axis([-1,1,-1,1])
    plt.title("Composition", fontsize = 10)
    plt.gca().set_aspect("equal")
    '''
    plt.show()
    return

# Problem 2
def solar_system(T, omega_e, omega_m):
    """Plot the trajectories of the earth and moon over the time interval [0,T]
    assuming the initial position of the earth is (x_e,0) and the initial
    position of the moon is (x_m,0).

    Parameters:
        T (int): The final time.
        x_e (float): The earth's initial x coordinate.
        x_m (float): The moon's initial x coordinate.
        omega_e (float): The earth's angular velocity.
        omega_m (float): The moon's angular velocity.
    """
    raise NotImplementedError("Problem 2 Incomplete")


def random_vector(n):
    """Generate a random vector of length n as a list."""
    return [random() for i in range(n)]

def random_matrix(n):
    """Generate a random nxn matrix as a list of lists."""
    return [[random() for j in range(n)] for i in range(n)]

def matrix_vector_product(A, x):
    """Compute the matrix-vector product Ax as a list."""
    m, n = len(A), len(x)
    return [sum([A[i][k] * x[k] for k in range(n)]) for i in range(m)]

def matrix_matrix_product(A, B):
    """Compute the matrix-matrix product AB as a list of lists."""
    m, n, p = len(A), len(B), len(B[0])
    return [[sum([A[i][k] * B[k][j] for k in range(n)])
                                    for j in range(p) ]
                                    for i in range(m) ]

# Problem 3
def prob3():
    """Use time.time(), timeit.timeit(), or %timeit to time
    matrix_vector_product() and matrix-matrix-mult() with increasingly large
    inputs. Generate the inputs A, x, and B with random_matrix() and
    random_vector() (so each input will be nxn or nx1).
    Only time the multiplication functions, not the generating functions.

    Report your findings in a single figure with two subplots: one with matrix-
    vector times, and one with matrix-matrix times. Choose a domain for n so
    that your figure accurately describes the growth, but avoid values of n
    that lead to execution times of more than 1 minute.
    """
    raise NotImplementedError("Problem 3 Incomplete")


# Problem 4
def prob4():
    """Time matrix_vector_product(), matrix_matrix_product(), and np.dot().

    Report your findings in a single figure with two subplots: one with all
    four sets of execution times on a regular linear scale, and one with all
    four sets of exections times on a log-log scale.
    """
    raise NotImplementedError("Problem 4 Incomplete")
