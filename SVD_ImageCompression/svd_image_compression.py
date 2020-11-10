# solutions.py
"""Volume 1: The SVD and Image Compression.
Caelan Osman
Math 345 Sec 3
Nov. 9, 2020
"""

import numpy as np
from numpy import linalg as la
from scipy import linalg as spla

# Problem 1
def compact_svd(A, tol=1e-6):
    """Compute the truncated SVD of A.

    Parameters:
        A ((m,n) ndarray): The matrix (of rank r) to factor.
        tol (float): The tolerance for excluding singular values.

    Returns:
        ((m,r) ndarray): The orthonormal matrix U in the SVD.
        ((r,) ndarray): The singular values of A as a 1-D array.
        ((r,n) ndarray): The orthonormal matrix V^H in the SVD.
    """
    #get eigenvalues/vectors
    eig_vals, eig_vecs = la.eig(A.conj().T @ A)
    #compute singular values
    singular_vals = np.sqrt(eig_vals)
    #get sort indexing in descending order
    ranking = np.argsort(singular_vals)[::-1]

    #get the sorted singular vals and vectors assuming the
    #corresponding singular value is greater than the tolerance
    #note that V_1_T is tranposed already because of the list comprehension
    sigma = np.array([singular_vals[index] for index in ranking if singular_vals[index] > tol])
    V_1_T = np.array([eig_vecs[:, index] for index in ranking if singular_vals[index] > tol])
    #the rank will be the length of sigma
    rank = len(sigma)
    #construct U_1 using list comprehension remember to transpose it
    U_1 = np.array([A @ V_1_T[i, :] /sigma[i] for i in range(rank)]).T

    return U_1, sigma, V_1_T


# Problem 2
def visualize_svd(A):
    """Plot the effect of the SVD of A as a sequence of linear transformations
    on the unit circle and the two standard basis vectors.
    """
    raise NotImplementedError("Problem 2 Incomplete")


# Problem 3
def svd_approx(A, s):
    """Return the best rank s approximation to A with respect to the 2-norm
    and the Frobenius norm, along with the number of bytes needed to store
    the approximation via the truncated SVD.

    Parameters:
        A ((m,n), ndarray)
        s (int): The rank of the desired approximation.

    Returns:
        ((m,n), ndarray) The best rank s approximation of A.
        (int) The number of entries needed to store the truncated SVD.
    """
    raise NotImplementedError("Problem 3 Incomplete")


# Problem 4
def lowest_rank_approx(A, err):
    """Return the lowest rank approximation of A with error less than 'err'
    with respect to the matrix 2-norm, along with the number of bytes needed
    to store the approximation via the truncated SVD.

    Parameters:
        A ((m, n) ndarray)
        err (float): Desired maximum error.

    Returns:
        A_s ((m,n) ndarray) The lowest rank approximation of A satisfying
            ||A - A_s||_2 < err.
        (int) The number of entries needed to store the truncated SVD.
    """
    raise NotImplementedError("Problem 4 Incomplete")


# Problem 5
def compress_image(filename, s):
    """Plot the original image found at 'filename' and the rank s approximation
    of the image found at 'filename.' State in the figure title the difference
    in the number of entries used to store the original image and the
    approximation.

    Parameters:
        filename (str): Image file path.
        s (int): Rank of new image.
    """
    raise NotImplementedError("Problem 5 Incomplete")
