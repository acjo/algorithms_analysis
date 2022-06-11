# drazin.py
"""
The Drazin Inverse.
Caelan Osman
June 10, 2022
"""

from multiprocessing.sharedctypes import Value
import sys
import pandas as pd
import csv
import numpy as np
import networkx as nx
from scipy import linalg as la
import time


# Helper function for problems 1 and 2.
def index(A, tol=1e-5):
    """Compute the index of the matrix A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.

    Returns:
        k (int): The index of A.
    """

    # test for non-singularity
    if not np.isclose(la.det(A), 0, rtol=tol, atol=tol):
        return 0

    n = len(A)
    k = 1
    Ak = A.copy()
    while k <= n:
        r1 = np.linalg.matrix_rank(Ak)
        r2 = np.linalg.matrix_rank(np.dot(A,Ak))
        if r1 == r2:
            return k
        Ak = np.dot(A,Ak)
        k += 1

    return k


# Problem 1
def is_drazin(A, Ad, k):
    """Verify that a matrix Ad is the Drazin inverse of A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.
        Ad ((n,n) ndarray): A candidate for the Drazin inverse of A.
        k (int): The index of A.

    Returns:
        (bool) True of Ad is the Drazin inverse of A, False otherwise.
    """

    # conditions to check if Ad is a Draizin inverse of A with index k. 
    if np.allclose(A@Ad, Ad@A)\
        and np.allclose(np.linalg.matrix_power(A, k+1)@Ad, np.linalg.matrix_power(A, k))\
             and np.allclose(Ad@A@Ad, Ad): 
             return True

    return False

# Problem 2
def drazin_inverse(A, tol=1e-4):
    """Compute the Drazin inverse of A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.

    Returns:
       ((n,n) ndarray) The Drazin inverse of A.
    """
    # initialize necesary variables
    n, _ = A.shape
    U = np.zeros((n, n))
    Z = np.zeros((n, n))
    # compute Schur decompositions
    T1, Q1, k1 = la.schur(A, sort = lambda x: np.abs(x) > tol)
    T2, Q2, k2 = la.schur(A, sort = lambda x: np.abs(x) <= tol)

    # populate change of basis matrix
    U[:, :k1] = Q1[:, :k1]
    U[:, k1:] = Q2[:, :n-k1]
    Uinv = la.inv(U)

    # Find T using the fact that T = U^(-1) @ A @ U
    V = Uinv @ A @ U

    # if k1 is zero then we know that A is nilpotent thus the draizin inverse is the zero matrix.
    if k1 != 0:
        Minv = la.inv(V[:k1, :k1])
        Z[:k1, :k1] = Minv

    return U @ Z @ Uinv


# Problem 3
def effective_resistance(A):
    """Compute the effective resistance for each node in a graph.

    Parameters:
        A ((n,n) ndarray): The adjacency matrix of an undirected graph.

    Returns:
        ((n,n) ndarray) The matrix where the ijth entry is the effective
        resistance from node i to node j.
    """

    # intialize necessary values
    n, _ = A.shape
    R = np.zeros((n, n))
    I = np.eye(n)
    # get Laplacian Matrix
    L = nx.laplacian_matrix(nx.from_numpy_array(A)).toarray()

    # compute effective resistence
    for j in range(n):
        Lj = L.copy()
        Lj[j] = I[j] 
        Ljd = drazin_inverse(Lj)
        R[:, j] = np.diag(Ljd)
        R[j, j] = 0

    return R 


# Problems 4 and 5
class LinkPredictor:
    """Predict links between nodes of a network."""

    def __init__(self, filename='social_network.csv'):
        """Create the effective resistance matrix by constructing
        an adjacency matrix.

        Parameters:
            filename (str): The name of a file containing graph data.
        """

        # with open (filename, "r") as in_file:
        #     data = in_file.read().replace('\n', ',').strip().split(',')
        # self.unique_names = np.unique(data)
        # print(self.unique_names)

        # read in as adata frame
        df = pd.read_csv(filename, header=None)
        # get all unique names
        self.unique_names = np.unique(df.values)
        # get name to index and index to name maps
        self.name_index_map = {name:i for i, name in enumerate(self.unique_names)}
        self.index_name_map = {i:name for i, name in enumerate(self.unique_names)}
        # get number of unique names
        self.n = len(self.unique_names)
        # populate adjacency matrix
        self.A = np.zeros((self.n, self.n))
        for i in range(df.shape[0]):
            names = df.iloc[i].values
            self.A[self.name_index_map[names[0]], self.name_index_map[names[1]]] += 1
            self.A[self.name_index_map[names[1]], self.name_index_map[names[0]]] += 1

        # calculate effective resistance matrix
        self.R = effective_resistance(self.A)

    def predict_link(self, node=None):
        """Predict the next link, either for the whole graph or for a
        particular node.

        Parameters:
            node (str): The name of a node in the network.

        Returns:
            node1, node2 (str): The names of the next nodes to be linked.
                Returned if node is None.
            node1 (str): The name of the next node to be linked to 'node'.
                Returned if node is not None.

        Raises:
            ValueError: If node is not in the graph.
        """

        if not node:

            # create copy of effective resistance matrix to modify
            Rmod = self.R.copy()
            # we want to find the lowest effective resistance of all nodes 
            # which are not yet connected
            mask = self.A > 0
            Rmod[mask] = np.inf
            # we cannot have self connections either. 
            np.fill_diagonal(Rmod, np.inf)
            
            # find index tuple of argmin
            ind = np.unravel_index(np.argmin(Rmod), Rmod.shape)
            
            # extract and return names
            name1 = self.index_name_map[ind[0]]
            name2 = self.index_name_map[ind[1]]
            return name1, name2

        else:
            # check that node is in graph
            if node not in self.unique_names:
                raise ValueError("{} is not in the graph.".format(node))

            # get column index
            col_index = self.name_index_map[node]

            # get column of A and R
            columnA = self.A[:, col_index]
            columnR = self.R[:, col_index].copy()

            # filter out nodes that already are linked
            mask = columnA > 0
            columnR[mask] = np.inf

            # no self linking
            columnR[col_index] = np.inf

            # get index and next link of lowest effective resistence for that node
            ind = np.argmin(columnR)
            next_link = self.index_name_map[ind]
            return next_link



    def add_link(self, node1, node2):
        """Add a link to the graph between node 1 and node 2 by updating the
        adjacency matrix and the effective resistance matrix.

        Parameters:
            node1 (str): The name of a node in the network.
            node2 (str): The name of a node in the network.

        Raises:
            ValueError: If either node1 or node2 is not in the graph.
        """

        # check that nodes are in the graph
        if node1 not in self.unique_names or node2 not in self.unique_names:
            raise ValueError("either {} or {} is not a node in the graph.".format(node1, node2))

        # get indices of nodes
        ind1 = self.name_index_map[node1]
        ind2 = self.name_index_map[node2]

        # update adjacency matrix
        self.A[ind1, ind2] += 1
        self.A[ind2, ind1] += 1
        # recalculate the effective resistance
        self.R = effective_resistance(self.A)


def main(key):

    if key == "1":
        A = np.array([[1, 3, 0, 0], [0, 1, 3, 0], [0, 0, 1, 3], [0, 0, 0, 0]])
        k = 1
        Ad = np.array([[1, -3, 9, 81], [0, 1, -3, -18], [0, 0, 1, 3], [0, 0, 0, 0]])

        assert is_drazin(A, Ad, k)

    elif key == "2":

        A = np.array([[1, 3, 0, 0], [0, 1, 3, 0], [0, 0, 1, 3], [0, 0, 0, 0]])
        kA = index(A)
        Ad = drazin_inverse(A)
        Ad_actual = np.array([[1, -3, 9, 81], [0, 1, -3, -18], [0, 0, 1, 3], [0, 0, 0, 0]])

        B = np.array([[1, 1, 3], [5, 2, 6], [-2, -1, -3]])
        kB = index(B)
        Bd = drazin_inverse(B)
        Bd_actual = np.zeros((3, 3))

        C = np.random.random(size=(12, 12))
        kC = index(A)
        Cd = drazin_inverse(C)

        assert np.allclose(Ad, Ad_actual)
        assert np.allclose(Bd, Bd_actual)
        assert is_drazin(A, Ad, kA)
        assert is_drazin(B, Bd, kB)
        assert is_drazin(C, Cd, kC)

    elif key == "3":

        #################
        # Graph:  
        # a--b--c--d
        #################
        A = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]])
        nodes = ["a", "b", "c", "d"]
        D = {node:i for i, node in enumerate(nodes)}
        R = effective_resistance(A)
        assert np.allclose(R[D["a"], D["c"]], 2)
        assert np.allclose(R[D["a"], D["d"]], 3)

        #################
        # Graph:  
        # a--b
        #################
        A = np.array([[0, 1], [1, 0]])
        nodes = ["a", "b",]
        D = {node:i for i, node in enumerate(nodes)}
        R = effective_resistance(A)
        assert np.allclose(R[D["a"], D["b"]], 1)

        #################
        # Graph:  
        #    c
        #   / \
        #  a---b
        #################
        A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        nodes = ["a", "b", "c"]
        D = {node:i for i, node in enumerate(nodes)}
        R = effective_resistance(A)
        assert np.allclose(R[D["a"], D["b"]], 2/3)
        assert np.allclose(R[D["a"], D["c"]], 2/3)

        #################
        # Graph:  
        #   / \
        #   a-b
        #   \ /
        #################
        A = 3*np.array([[0, 1], [1, 0]])
        nodes = ["a", "b",]
        D = {node:i for i, node in enumerate(nodes)}
        R = effective_resistance(A)
        assert np.allclose(R[D["a"], D["b"]], 1/3)

        #################
        # Graph:  
        #   / \
        #   a b
        #   \ /
        #################
        A = 2*np.array([[0, 1], [1, 0]])
        nodes = ["a", "b",]
        D = {node:i for i, node in enumerate(nodes)}
        R = effective_resistance(A)
        assert np.allclose(R[D["a"], D["b"]], 1/2)

        #################
        # Graph:  
        #   / \
        #    -
        #   a-b
        #    -
        #   \ /
        #################
        A = 4*np.array([[0, 1], [1, 0]])
        nodes = ["a", "b",]
        D = {node:i for i, node in enumerate(nodes)}
        R = effective_resistance(A)
        assert np.allclose(R[D["a"], D["b"]], 1/4)

    elif key == "4":
        LP = LinkPredictor()
        assert np.any(LP.A != np.zeros(LP.A.shape))

    elif key ==  "5":
        LP = LinkPredictor()
        name1, name2 = LP.predict_link(node=None)
        assert name1 == "Oliver"
        assert name2 == "Emily"

        LP = LinkPredictor()
        next_link = LP.predict_link(node="Melanie")
        assert next_link == "Carol"

        LP = LinkPredictor()
        next_link = LP.predict_link(node="Alan")
        assert next_link == "Sonia"
        LP.add_link("Alan", "Sonia")

        next_link = LP.predict_link(node="Alan")
        assert next_link == "Piers"
        LP.add_link("Alan", "Piers")
        next_link = LP.predict_link(node="Alan")
        assert next_link == "Abigail"

        try:
            LP.add_link("Alan", "Caelan")
        except ValueError:
            assert True
        else:
            print("No ValueError exception raised.")
            assert False

    elif key == "all":
        main("1")
        main("2")
        main("3")
        main("4")
        main("5")

    else:
        raise ValueError("Incorrect problem specification.")

        

    return

if __name__ == "__main__":

    if len(sys.argv) == 2:
        main(sys.argv[1])

