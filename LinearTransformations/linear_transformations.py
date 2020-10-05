# linear_transformations.py
"""Volume 1: Linear Transformations.
Caelan Osman
Math 345 Sec 3
September 29, 2020
"""

from random import random
import numpy as np
from matplotlib import pyplot as plt
import time


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
    '''
       This function plots a bunch of the linear transformations against the standards 
    '''
    data = np.load('horse.npy')

    #figure
    ax1 = plt.subplot(231)
    ax1.plot(data[0], data[1], 'k,')
    plt.axis([-1,1,-1,1])
    plt.title("Original", fontsize = 10)
    plt.gca().set_aspect("equal")

    ax2 = plt.subplot(232)
    stretch_trans = stretch(data, 1/2, 6/5)
    ax2.plot(stretch_trans[0], stretch_trans[1], 'k,')
    plt.axis([-1,1,-1,1])
    plt.title("Stretch", fontsize = 10)
    plt.gca().set_aspect("equal")


    ax3 = plt.subplot(233)
    shear_trans = shear(data, 1/2, 0)
    ax3.plot(shear_trans[0], shear_trans[1], 'k,')
    plt.axis([-1,1,-1,1])
    plt.title("Shear", fontsize = 10)
    plt.gca().set_aspect("equal")

    ax4 = plt.subplot(234)
    reflect_trans = reflect(data, 0, 1)
    ax4.plot(reflect_trans[0], reflect_trans[1], 'k,')
    plt.axis([-1,1,-1,1])
    plt.title("Reflection", fontsize = 10)
    plt.gca().set_aspect("equal")

    ax5 = plt.subplot(235)
    rotate_trans = rotate(data, np.pi/2)
    ax5.plot(rotate(data, np.pi/2)[0], rotate(data, np.pi/2)[1], 'k,')
    plt.axis([-1,1,-1,1])
    plt.title("Rotation", fontsize = 10)
    plt.gca().set_aspect("equal")

    ax6 = plt.subplot(236)
    composition_trans = rotate(reflect(shear(stretch(data, 1/2, 6/5), 1/2, 0), 0, 1), np.pi/2)
    ax6.plot(composition_trans[0], composition_trans[1], 'k,')
    plt.axis([-1,1,-1,1])
    plt.title("Composition", fontsize = 10)
    plt.gca().set_aspect("equal")
    plt.show()
    return

# Problem 2
def solar_system(T, x_e, x_m, omega_e, omega_m):
    """This function plots the trajectories of the earth and moon over the time interval [0,T]
    assuming the initial position of the earth is (x_e,0) and the initial
    position of the moon is (x_m,0).
    """
    #set the intervals of time
    times = np.linspace(0, T, 500)
    rads_earth = times * omega_e
    rads_moon = times * omega_m
    #initialize earth and moon initial poisitions and relative position of moon to earth
    initial_earth = np.array([[x_e], [0]])
    initial_moon = np.array([[x_m], [0]])
    relative_pos = initial_moon - initial_earth
    #get x, y position of earth and moon by using our rotate function from problem 1, use list comprehension
    #to make graphing easier
    pos_earth_x = [rotate(initial_earth, rads)[0] for rads in rads_earth]
    pos_earth_y = [rotate(initial_earth, rads)[1] for rads in rads_earth]
    pos_moon_x = [rotate(relative_pos, rads_moon[i])[0] + pos_earth_x[i] for i in range(0, 500)]
    pos_moon_y = [rotate(relative_pos, rads_moon[i])[1] + pos_earth_y[i] for i in range(0, 500)]

    #plot the functions
    plt.plot(pos_earth_x, pos_earth_y, label = 'Earth')
    plt.plot(pos_moon_x, pos_moon_y, label = 'Moon')
    plt.axis('equal')
    plt.legend(loc = 'lower right')
    plt.show()
    return

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
    '''This funciton generates the amount of time calculations for matrix-vector and matrix-matrix
       multiplication takes with an increasing size of matrix n'''
    #vectors containing the dimension of the vector/matrices and
    #the times to compute matrix vector and matrix matrix multiplication
    sizes = 2**np.arange(1,8)
    time_m_vec = []
    time_m_m = []

    #for loop to calculate times
    for n in sizes:
        vec_x = random_vector(n)
        mat_A = random_matrix(n)
        mat_B = random_matrix(n)
        m_vec_start = time.time()
        matrix_vector_product(mat_A, vec_x)
        m_vec_end = time.time()
        m_m_start = time.time()
        matrix_matrix_product(mat_A, mat_B)
        m_m_end = time.time()
        time_m_vec.append(m_vec_end - m_vec_start)
        time_m_m.append(m_m_end - m_m_start)

    #plot sizes against the times
    ax1 = plt.subplot(121)
    ax1.plot(sizes, time_m_vec, '.-', linewidth=2, markersize=10)
    plt.title("Matrix-Vector Multiplication", fontsize = 10)
    plt.xlabel('n')
    plt.ylabel('Seconds')

    ax2 = plt.subplot(122)
    ax2.plot(sizes, time_m_m, 'g.-', linewidth=2, markersize=10)
    plt.title("Matrix-Matrix Multiplication", fontsize = 10)
    plt.xlabel('n')
    plt.ylabel('Seconds')
    plt.show()
    return

# Problem 4
def prob4():
    """This function compares our computation times with the built in matrix-vector and matrix-matrix
       multiplication isnside NumPy. We plot these and compare them on both a linear and logarithmic scale.
    """
    #array and list holding size and the respective times
    sizes = 2**np.arange(1,8)
    time_m_vec = []
    time_m_m = []
    time_m_vec_np = []
    time_m_m_np = []

    #for loop to calculate times
    for n in sizes:
        vec_x = random_vector(n)
        mat_A = random_matrix(n)
        mat_B = random_matrix(n)
        m_vec_start = time.time()
        matrix_vector_product(mat_A, vec_x)
        m_vec_end = time.time()
        m_m_start = time.time()
        matrix_matrix_product(mat_A, mat_B)
        m_m_end = time.time()
        time_m_vec.append(m_vec_end - m_vec_start)
        time_m_m.append(m_m_end - m_m_start)
        m_vec_start = time.time()
        np.array(mat_A) @ np.array(vec_x)
        m_vec_end = time.time()
        m_m_start = time.time()
        np.array(mat_A) @ np.array(mat_B)
        m_m_end = time.time()
        time_m_vec_np.append(m_vec_end - m_vec_start)
        time_m_m_np.append(m_m_end - m_m_start)

    #making linear subplot
    ax1 = plt.subplot(121)
    ax1.plot(sizes, time_m_vec, 'b.-', lw=2, ms=15, label='matrix-vector')
    ax1.plot(sizes, time_m_m, 'g.-', lw=2, ms=15, label='matrix-matrix')
    ax1.plot(sizes, time_m_vec_np, 'r.-', lw=2, ms = 15, label='matrix-vector-np')
    ax1.plot(sizes, time_m_m_np, 'c.-', lw=2, ms=15, label='matrix-matrix-np')
    ax1.set_title("Linear", fontsize=18)
    ax1.legend(loc='upper left')

    #making logarithmic subplot
    ax2 = plt.subplot(122)
    ax2.loglog(sizes, time_m_vec, 'b.-', lw=2, ms=15, label='matrix-vector', basex=2, basey=2)
    ax2.loglog(sizes, time_m_m, 'g.-', lw=2, ms=15, label='matrix-matrix', basex=2, basey=2)
    ax2.loglog(sizes, time_m_vec_np, 'r.-', lw=2, ms = 15, label='matrix-vector-np', basex=2, basey=2)
    ax2.loglog(sizes, time_m_m_np, 'c.-', lw=2, ms=15, label='matrix-matrix-np', basex=2, basey=2)
    ax2.set_title("Logarithmic", fontsize=18)
    ax2.legend(loc='upper left')

    plt.show()
    return
