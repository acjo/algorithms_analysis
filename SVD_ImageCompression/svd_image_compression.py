# solutions.py
"""Volume 1: The SVD and Image Compression.
Caelan Osman
Math 345 Sec 3
Nov. 9, 2020
"""

import numpy as np
from numpy import linalg as la
from scipy import linalg as spla
from matplotlib import pyplot as plt
from imageio import imread

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
    singular_vals = np.sqrt(abs(eig_vals))
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
    #create x and y coordinates
    theta = np.linspace(0, 2 * np.pi, 200)
    x = np.cos(theta)
    y = np.sin(theta)

    #create S
    S = np.vstack((x, y))
    #create E
    E = np.array([[1, 0, 0], [0, 0, 1]])
    #compute SVD
    U, sigma, Vh = spla.svd(A, full_matrices=True)

    #plot original
    ax1 = plt.subplot(221)
    ax1.plot(x, y)
    ax1.plot(E[0, :], E[1, :])
    ax1.set_title('Original')
    plt.axis('equal')

    #plot second operation
    operation_1_S = Vh @ S
    operation_1_E = Vh @ E
    ax2 = plt.subplot(222)
    ax2.plot(operation_1_S[0, :], operation_1_S[1, :])
    ax2.plot(operation_1_E[0, :], operation_1_E[1, :])
    ax2.set_title('1st Operation')
    plt.axis('equal')

    #plot 3rd operation
    operation_2_S = np.diag(sigma) @ operation_1_S
    operation_2_E = np.diag(sigma) @ operation_1_E
    ax3 = plt.subplot(223)
    ax3.plot(operation_2_S[0, :], operation_2_S[1, :])
    ax3.plot(operation_2_E[0, :], operation_2_E[1, :])
    ax3.set_title('2nd Operation')
    plt.axis('equal')

    #plot final operation
    operation_3_S = U @ operation_2_S
    operation_3_E = U @ operation_2_E
    ax4 = plt.subplot(224)
    ax4.plot(operation_3_S[0, :], operation_3_S[1, :])
    ax4.plot(operation_3_E[0, :], operation_3_E[1, :])
    ax4.set_title('3rd Operation')
    plt.axis('equal')

    plt.show()


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
    #gets the compact svd, S will be contain all nonzero singularvalues
    u, S, vh = spla.svd(A, full_matrices=False)

    #if s is too large, throw an error
    if s > la.matrix_rank(A, 1e-8):
        raise ValueError(str(s) + ' is larger than the rank of the matrix')

    #create the truncated svd. the function deletes the colums (or rows) from index s to the end
    u_c = np.delete(u, np.s_[s:], 1)
    S_c = np.delete(S, np.s_[s:])
    vh_c = np.delete(vh, np.s_[s:], 0)


    return u_c @ np.diag(S_c) @ vh_c, u_c.size + S_c.size + vh_c.size


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
    #get compact svd
    u, S, vh = spla.svd(A, full_matrices=False)

    #S is given in non-increasing order so the smallest element of S
    #will be the last. We need to make sure the tolerance isn't too small
    if err <= S[-1]:
        raise ValueError('The error is too small')

    #getting the largest singular value that
    #is smaller than the error
    i = len(S) - 1
    val = S[-1]
    while i >= 0:
        i -= 1
        #if the current value is less than the error try the next value
        if val < err:
            val = S[i]
        #otherwise break, i will be the index we need to start delete at so add 1 back for the
        #current loop decrement, and an additional 1 because that's where we need to start deleting
        else:
            i += 2
            break

    #set the low rank approximations
    u_c = np.delete(u, np.s_[i:], 1)
    S_c = np.delete(S, np.s_[i:])
    vh_c = np.delete(vh, np.s_[i:], 0)

    #return A_s and the number of elements in the best rank approximation SVD
    return u_c @ np.diag(S_c) @ vh_c, u_c.size + S_c.size + vh_c.size

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
    #read in image
    scaled = imread(filename) / 255
    size = scaled.size

    #check if the image is grayscale
    if scaled.ndim == 3:
        #get all the color layers
        R = scaled[:, :, 0]
        G = scaled[:, :, 1]
        B = scaled[:, :, 2]

        #get low rank approximations and elements required
        Rs, element_r = svd_approx(R, s)
        Gs, element_g = svd_approx(G, s)
        Bs, element_b = svd_approx(B, s)

        #stack the layers back together
        #stacked = np.dstack(Rs, Gs, Bs)
        #make sure all elements are in [0, 1]
        #image_s = np.where(np.where(stacked > 1, 1, stacked) < 0, 0, np.where(stacked > 1, 1, stacked))
        image_s = np.clip(np.dstack((Rs, Gs, Bs)), 0, 1)

        #plot
        ax1 = plt.subplot(121)
        ax1.imshow(scaled)
        ax1.set_title('Original')
        ax1.axis('off')

        ax2 = plt.subplot(122)
        ax2.imshow(image_s)
        ax2.set_title('Low Rank ' + str(s) + ' Approximation')
        ax2.axis('off')

        plt.suptitle('Element difference: ' + str(size - element_r - element_g - element_b))

        plt.show()


    else:
        #get low rank approximation
        low_rank_approx, element = svd_approx(scaled, s)
        #make sure all elements are in [0, 1]
        image_s = np.clip(low_rank_approx, 0, 1)

        #plot
        ax1 = plt.subplot(121)
        ax1.imshow(scaled, cmap='gray')
        ax1.set_title('Original')
        ax1.axis('off')

        ax2 = plt.subplot(122)
        ax2.imshow(image_s, cmap='gray')
        ax2.set_title('Low Rank ' + str(s) + ' Approximation')
        ax2.axis('off')

        plt.suptitle('Element difference: ' + str((size - element)))

        plt.show()
