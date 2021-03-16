# pagerank.py
"""Volume 1: The Page Rank Algorithm.
Caelan Osman
Math 322 Sec. 2
March 9th, 2021
"""
import time
import numpy as np
from scipy import linalg as la

# Problems 1-2
class DiGraph:
    """A class for representing directed graphs via their adjacency matrices.

    Attributes:
        self.A the adjacency matrix
        self.n the size of the nxn A matrix
        self.labels the labels of the graph
    """
    # Problem 1
    def __init__(self, A, labels=None):
        """Modify A so that there are no sinks in the corresponding graph,
        then calculate Ahat. Save Ahat and the labels as attributes.

        Parameters:
            A ((n,n) ndarray): the adjacency matrix of a directed graph.
                A[i,j] is the weight of the edge from node j to node i.
            labels (list(str)): labels for the n nodes in the graph.
                If None, defaults to [0, 1, ..., n-1].
        """

        self.n = A.shape[0]
        #check and ccount for sinks
        for i in range(self.n):
            if np.all(A[:, i] == 0):
                A[:, i] = 1

        #use array broadcasting to deal with boredom
        norms = np.array([1 / la.norm(A[:, i], ord=1) for i in range(self.n)])
        self.A = A * norms

        #save labels as an attribute
        if labels is None:
            self.labels = [str(i) for i in range(self.n)]
        else:
            if len(labels) != self.n:
                raise ValueError('Number of nodes and labels need to be the same.')
            self.labels = labels

    # Problem 2
    def linsolve(self, epsilon=0.85):
        """Compute the PageRank vector using the linear system method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Returns:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        #use a linear solver to get the page rank
        I = np.eye(self.n)
        p = la.solve(I - epsilon*self.A, (1-epsilon) /self.n * np.ones(self.n))
        page_rank = {label : p[i] for i, label in enumerate(self.labels)}

        return page_rank

    # Problem 2
    def eigensolve(self, epsilon=0.85):
        """Compute the PageRank vector using the eigenvalue method.
        Normalize the resulting eigenvector so its entries sum to 1.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        #solve an eigenvalue problem to get the page rank
        B = epsilon * self.A + ((1 - epsilon) / self.n) * np.ones(self.A.shape)
        #get eigenvalues and eigenvectors
        vals_vecs = la.eig(B)
        #find index of val/vec where val = 1 (guranteed to be largest)
        index = np.argmax(vals_vecs[0])
        #get page rank vector
        p = vals_vecs[1][:, 0]
        #normalize
        p /= p.sum()
        #get page rank
        page_rank = {label : p[i] for i, label in enumerate(self.labels)}

        return page_rank


    # Problem 2
    def itersolve(self, epsilon=0.85, maxiter=100, tol=1e-12):
        """Compute the PageRank vector using the iterative method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.
            maxiter (int): the maximum number of iterations to compute.
            tol (float): the convergence tolerance.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values. """
        #create intial vector
        p0 = np.ones(self.n) / self.n
        #get next iteration
        p1 = epsilon * self.A @ p0 + ((1-epsilon) / self.n) * np.ones(self.n)
        #create iteration count
        i = 1

        #compute until maxing out maxiter or we are within the tolerance
        while i < maxiter:
            p0 = p1
            p1 = epsilon * self.A @ p0 + ((1-epsilon) / self.n) * np.ones(self.n)
            if la.norm(p1 - p0, ord=1) < tol:
                break
            i += 1
        page_rank = {label : p1[i] for i, label in enumerate(self.labels)}
        return page_rank

# Problem 3
def get_ranks(d):
    """Construct a sorted list of labels based on the PageRank vector.

    Parameters:
        d (dict(str -> float)): a dictionary mapping labels to PageRank values.

    Returns:
        (list) the keys of d, sorted by PageRank value from greatest to least.
    """
    #keys = list(d.keys())
    vals = list(d.values())
    #first method for sorting
    #sorted_labels = labels = [label for _, label in sorted(zip(vals,keys), key=lambda pair: pair[0], reverse=True)]
    #another method for sorting is the following
    #import operator
    #sorted_dict = sorted(my_dict.items(), key=operator.itemgetter(1), reverse=True)
    sorted_labels = [first for first, _ in sorted(d.items(), key=lambda item: item[1], reverse=True)]
    return sorted_labels

# Problem 4
def rank_websites(filename="web_stanford.txt", epsilon=0.85):
    """Read the specified file and construct a graph where node j points to
    node i if webpage j has a hyperlink to webpage i. Use the DiGraph class
    and its itersolve() method to compute the PageRank values of the webpages,
    then rank them with get_ranks(). If two webpages have the same rank,
    resolve ties by listing the webpage with the larger ID number first.

    Each line of the file has the format
        a/b/c/d/e/f...
    meaning the webpage with ID 'a' has hyperlinks to the webpages with IDs
    'b', 'c', 'd', and so on.

    Parameters:
        filename (str): the file to read from.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of webpage IDs.
    """
    #get file content
    with open(filename) as infile:
        content = infile.readlines()
    
    #get each line in array
    lines = [line.strip().split("/") for line in content]
    #get only ids that have their own line
    listed_ids = np.array([int(line[0]) for line in lines])
    #get dictionary of listed ids that map to their respective lines
    index_line = {int(line[0]): line[1:] for line in lines}
    #get all ids in a list including those who don't have a line (put in set first to elminate duplicates)
    total_ids = np.array(list({int(line[i]) for line in lines for i in range(len(line))}))
    #get total number of ids
    n = total_ids.size
    #we now need to sort in ascending order
    ordering = np.argsort(total_ids)
    #get ordered labels
    ordered_labels = total_ids[ordering]
    #create index mapping i.e label -> index (will be used later)
    index_mapping = {label : i for i, label in enumerate(ordered_labels)}
    #create empty adjacency matrix of zeros
    A = np.zeros((n, n))
    #now fill adjacency matrix
    for id in listed_ids:
        column_id = index_mapping[id]
        mapped_ids = index_line[id]
        for linked in mapped_ids:
            row_id = index_mapping[int(linked)]
            A[row_id, column_id] = 1
    
    #create graph class
    graph = DiGraph(A, labels=ordered_labels.astype(str))
    #get the ranking dictionary using itersolve
    ranking = graph.itersolve(epsilon=epsilon)
    #get and return sorted_rankings
    return get_ranks(ranking)


# Problem 5
def rank_ncaa_teams(filename, epsilon=0.85):
    """Read the specified file and construct a graph where node j points to
    node i with weight w if team j was defeated by team i in w games. Use the
    DiGraph class and its itersolve() method to compute the PageRank values of
    the teams, then rank them with get_ranks().

    Each line of the file has the format
        A,B
    meaning team A defeated team B.

    Parameters:
        filename (str): the name of the data file to read.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of team names.
    """
    raise NotImplementedError("Problem 5 Incomplete")


# Problem 6
def rank_actors(filename="top250movies.txt", epsilon=0.85):
    """Read the specified file and construct a graph where node a points to
    node b with weight w if actor a and actor b were in w movies together but
    actor b was listed first. Use NetworkX to compute the PageRank values of
    the actors, then rank them with get_ranks().

    Each line of the file has the format
        title/actor1/actor2/actor3/...
    meaning actor2 and actor3 should each have an edge pointing to actor1,
    and actor3 should have an edge pointing to actor2.
    """
    raise NotImplementedError("Problem 6 Incomplete")


if __name__ == "__main__":
    #problem 1 & 2:
    '''
    A = np.array([[0, 0, 0, 0],
                  [1, 0, 1, 0],
                  [1, 0, 0, 1],
                  [1, 0, 1, 0]])

    final_A = np.array([[0, 1/4., 0, 0],
                      [1/3., 1/4., 1/2., 0],
                      [1/3., 1/4., 0, 1],
                      [1/3., 1/4., 1/2., 0]])
    
    labels = ["a", "b", "c", "d"]
    G = DiGraph(A, labels)

    print(np.allclose(G.A, final_A))

    final_dict = {'a': 0.095758635, 'b': 0.274158285, 'c': 0.355924792, 'd': 0.274158285}
    #test problem 2
    #linsolve test
    lin = G.linsolve()
    equiv_lin = np.array([np.allclose(final_dict[key], lin[key]) for key in final_dict])
    print(np.all(equiv_lin))

    #eigensolve test
    eig = G.eigensolve()
    equiv_eig = np.array([np.allclose(final_dict[key], eig[key]) for key in final_dict])
    print(np.all(equiv_eig))

    #itersolve test
    iters = G.itersolve()
    equiv_iters = np.array([np.allclose(final_dict[key], iters[key]) for key in final_dict])
    print(np.all(equiv_iters))

    #test problem 3
    possible_1 = ['c', 'b', 'd', 'a']
    possible_2 = ['c', 'd', 'b', 'a']
    rank = get_ranks(iters)
    print(rank)
    truth = (np.all(possible_1 == rank) or np.all(possible_2 == rank))
    print(truth)
    '''
    #problem 4
    #print(np.all(rank_websites()[:3] == ['98595', '32791', '28392']))