# montecarlo_integration.py
"""Volume 1: Monte Carlo Integration.
Caelan Osman
Math 347, Sec. 2
Feb. 22nd, 2021
"""

import numpy as np
from scipy import stats
from scipy import linalg as la
from matplotlib import pyplot as plt

# Problem 1
def ball_volume(n, N=10000):
    """Estimate the volume of the n-dimensional unit ball.

    Parameters:
        n (int): The dimension of the ball. n=2 corresponds to the unit circle,
            n=3 corresponds to the unit sphere, and so on.
        N (int): The number of random points to sample.

    Returns:
        (float): An estimate for the volume of the n-dimensional unit ball.
    """
    #get an nxN dimensional array of sample points form the uniform distribution
    sample_points = np.random.uniform(-1, 1, (n, N))
    #get the lengths with the 2 norm
    lengths = la.norm(sample_points, axis=0, ord=2)
    #get the number of points within the unitball
    num_within = np.count_nonzero(lengths < 1)
    #return the percentage of points within the ball multiplied by the hypervolume of the square
    return 2**n * num_within / N


# Problem 2
def mc_integrate1d(f, a, b, N=10000):
    """Approximate the integral of f on the interval [a,b].

    Parameters:
        f (function): the function to integrate. Accepts and returns scalars.
        a (float): the lower bound of interval of integration.
        b (float): the lower bound of interval of integration.
        N (int): The number of random points to sample.

    Returns:
        (float): An approximation of the integral of f over [a,b].

    Example:
        >>> f = lambda x: x**2
        >>> mc_integrate1d(f, -4, 2)    # Integrate from -4 to 2.
        23.734810301138324              # The true value is 24.
    """
    #get sample points
    sample_points = np.random.uniform(a, b, N)
    #return the length of the interval multpilied by the sum of the
    #outputs of the smaple points divided by the number of points
    return (b-a) * np.sum(f(sample_points)) / N


# Problem 3
def mc_integrate(f, mins, maxs, N=10000):
    """Approximate the integral of f over the box defined by mins and maxs.

    Parameters:
        f (function): The function to integrate. Accepts and returns
            1-D NumPy arrays of length n.
        mins (list): the lower bounds of integration.
        maxs (list): the upper bounds of integration.
        N (int): The number of random points to sample.

    Returns:
        (float): An approximation of the integral of f over the domain.

    Example:
        # Define f(x,y) = 3x - 4y + y^2. Inputs are grouped into an array.
        >>> f = lambda x: 3*x[0] - 4*x[1] + x[1]**2

        # Integrate over the box [1,3]x[-2,1].
        >>> mc_integrate(f, [1, -2], [3, 1])
        53.562651072181225              # The true value is 54.
    """
    #get the dimension of the space
    n = len(mins)
    #get the sample points
    sample_points = np.random.uniform(0, 1, (n, N))

    #get the volume of the box
    V_Omega = np.prod(np.array([b - mins[i] for i, b in enumerate(maxs)]))

    #scale box and shift box
    for i, a in enumerate(mins):
        b = maxs[i]
        sample_points[i] *= (b - a)
        sample_points[i] += a

    #get evaluation at the N columns
    evaluation = sum([f(sample_points[:, i]) for i in range(N)])

    #return the average
    return V_Omega * evaluation / N



# Problem 4
def prob4():
    """Let n=4 and Omega = [-3/2,3/4]x[0,1]x[0,1/2]x[0,1].
    - Define the joint distribution f of n standard normal random variables.
    - Use SciPy to integrate f over Omega.
    - Get 20 integer values of N that are roughly logarithmically spaced from
        10**1 to 10**5. For each value of N, use mc_integrate() to compute
        estimates of the integral of f over Omega with N samples. Compute the
        relative error of estimate.
    - Plot the relative error against the sample size N on a log-log scale.
        Also plot the line 1 / sqrt(N) for comparison.
    """

    #get min values
    min_values = np.array([-3/2., 0, 0, 0])
    #get max values
    max_values = np.array([3/4., 1, 1/2., 1])
    #initialize f
    fx = lambda x: (1/ (2*np.pi**2)) *np.exp(-np.inner(x, x))
    #comparison function
    c = lambda n : 1 / np.sqrt(n)
    #get logspaced values
    n_vals = np.logspace(1, 5, 20)
    n_vals = np.array(n_vals, dtype=np.int)
    #get approximate from problem 3 function
    approximate = np.array([mc_integrate(fx, min_values, max_values, n) for n in n_vals])
    #set mean values and covariance matrix
    means, cov = np.zeros(4), np.eye(4)
    #get "exact" value
    exact = stats.mvn.mvnun(min_values, max_values, means, cov)[0]
    #get relative error values
    relative_error = np.array([np.abs(exact - approximate[n]) / np.abs(exact) for n in range(20)])
    #plot on log scale
    plt.loglog(n_vals, relative_error, 'go-', markersize=3, label='Error')
    plt.loglog(n_vals, c(n_vals), 'co-', markersize=3, label='1/sqr(n)')
    plt.legend(loc='best')
    plt.show()

if __name__ == "__main__":

    #problem 1
    '''
    estimated_volumes = {n: ball_volume(n, N=1000000) for n in range(3, 10, 2)}
    print(estimated_volumes)
    '''
    #problem 2
    '''
    f = lambda x: x**2
    print(mc_integrate1d(f, -4, 2))
    '''

    #problem 3
    #test 1
    '''
    fx = lambda x: x[0]**2 + x[0]**2
    min_vals = [0, 0]
    max_vals = [1, 1]

    print(mc_integrate(fx, min_vals, max_vals))
    '''

    '''
    fx = lambda x: 3*x[0] - 4*x[1] + x[1]**2
    min_vals = [1, -2]
    max_vals = [3, 1]
    print(mc_integrate(fx, min_vals, max_vals, 100000))
    '''

    '''
    fx = lambda x: x[0] + x[1] - x[3]*x[2]**2
    min_vals = [-1, -2, -3, -4]
    max_vals = [1, 2, 3, 4]
    print(mc_integrate(fx, min_vals, max_vals, 100000000))
    '''

    #problem 4
    #prob4()


