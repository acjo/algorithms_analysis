#image_segmentation.py
"""Volume 1: Image Segmentation.
Caelan Osman
Math 345, Sec 3
November 2, 2020
"""

import numpy as np
from scipy import sparse
from scipy import linalg as la
from matplotlib import pyplot as plt
from imageio import imread

# Problem 1
def laplacian(A):
    """Compute the Laplacian matrix of the graph G that has adjacency matrix A.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.

    Returns:
        L ((N,N) ndarray): The Laplacian matrix of G.
    """
    D = np.diag(A.sum(axis = 0))
    return D - A

# Problem 2
def connectivity(A, tol=1e-8):
    """Compute the number of connected components in the graph G and its
    algebraic connectivity, given the adjacency matrix A of G.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.
        tol (float): Eigenvalues that are less than this tolerance are
            considered zero.

    Returns:
        (int): The number of connected components in G.
        (float): the algebraic connectivity of G.
    """
    #get the real part of the eigenvalues
    eigen_values = la.eigvals(laplacian(A)).real
    #create the mask
    mask = eigen_values < tol
    #get the zero eigenvalues
    zero_eigen_values = eigen_values[mask]
    #get the number of connected componnents in the graph
    num_connected = zero_eigen_values.size
    #get the minimum eigenvalue
    minimum = min(eigen_values)

    #if the number of connected components is greater than or equal to 2 then the connectivity is
    #obviously zero.
    if num_connected >= 2:
        connectivity = 0
    #otherwise loop through the array and get the second smallest eigenvalue (which will have to be larger than zero)
    else:
        connectivity = max(eigen_values)
        for i in range(0, eigen_values.size):
            if eigen_values[i] < connectivity and eigen_values[i] > minimum:
                connectivity = eigen_values[i]

    return num_connected, connectivity


# Helper function for problem 4.
def get_neighbors(index, radius, height, width):
    """Calculate the flattened indices of the pixels that are within the given
    distance of a central pixel, and their distances from the central pixel.

    Parameters:
        index (int): The index of a central pixel in a flattened image array
            with original shape (radius, height).
        radius (float): Radius of the neighborhood around the central pixel.
        height (int): The height of the original image in pixels.
        width (int): The width of the original image in pixels.

    Returns:
        (1-D ndarray): the indices of the pixels that are within the specified
            radius of the central pixel, with respect to the flattened image.
        (1-D ndarray): the euclidean distances from the neighborhood pixels to
            the central pixel.
    """
    # Calculate the original 2-D coordinates of the central pixel.
    row, col = index // width, index % width

    # Get a grid of possible candidates that are close to the central pixel.
    r = int(radius)
    x = np.arange(max(col - r, 0), min(col + r + 1, width))
    y = np.arange(max(row - r, 0), min(row + r + 1, height))
    X, Y = np.meshgrid(x, y)

    # Determine which candidates are within the given radius of the pixel.
    R = np.sqrt(((X - col)**2 + (Y - row)**2))
    mask = R < radius
    return (X[mask] + Y[mask]*width).astype(np.int), R[mask]


# Problems 3-6
class ImageSegmenter:
    """Class for storing and segmenting images."""

    # Problem 3
    def __init__(self, filename):
        """Read the image file. Store its brightness values as a flat array."""
        #load the image, scale it, set is an attribute
        self.scaled = imread(filename) / 255
        #if there are 3 axes and the 3rd axes has size 3 then set the brightness
        #by averaging the RGB values otherwise the image is the brightness
        size = self.scaled.shape
        if len(size) == 3:
            if size[2] == 3:
                self.rgb = True
                self.brightness = self.scaled.mean(axis=2)
        else:
            self.rgb = False
            self.brightness = self.scaled

        #unravel the brigthness array
        self.flatten = np.ravel(self.brightness)


    # Problem 3
    def show_original(self):
        """Display the original image."""
        if self.rgb:
            plt.imshow(self.scaled)
            plt.axis('off')
            plt.show() #TODO: Find out if we are supposed to include this?
        else:
            plt.imshow(self.scaled, cmap='gray')
            plt.axis('off')
            plt.show() #TODO: Find out if we are supposed to include this?

    # Problem 4
    def adjacency(self, r=5., sigma_B2=.02, sigma_X2=3.):
        """Compute the Adjacency and Degree matrices for the image graph."""

    # Problem 5
    def cut(self, A, D):
        """Compute the boolean mask that segments the image."""
        #compute the laplacian
        L = sparse.csgraph.laplacian(A)
        #construct D^(-1/2)
        D_1_2 = sparse.diags(D**(-1/2))
        #construct D^(-1/2)LD^(-1/2)
        cut = D_1_2 @ L @ D_1_2
        #compute the eigenvector corresponding to the smallest eigenvalue
        eigen_vec = sparse.linalg.eigsh(which='SM', k=2)

    # Problem 6
    def segment(self, r=5., sigma_B=.02, sigma_X=3.):
        """Display the original image and its segments."""
        A, D = self.adjacency(r, sigma_B, sigma_X)
        mask = self.cut(A, D)


A = np.array([[1,2,3], [4,5,6]]).astype(np.float64)
print(A**(-1/2))

# if __name__ == '__main__':
#     ImageSegmenter("dream_gray.png").segment()
#     ImageSegmenter("dream.png").segment()
#     ImageSegmenter("monument_gray.png").segment()
#     ImageSegmenter("monument.png").segment()
