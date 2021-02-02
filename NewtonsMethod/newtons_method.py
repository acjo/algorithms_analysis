# newtons_method.py
"""Volume 1: Newton's Method.
Caelan osman
Math 347 Sec. 2
Jan. 29, 2021
"""

import numpy as np
from numpy import linalg as la
from scipy.optimize import newton as nwton
from matplotlib import pyplot as plt

# Problems 1, 3, and 5
def newton(f, x0, Df, tol=1e-5, maxiter=15, alpha=1.):
    """Use Newton's method to approximate a zero of the function f.

    Parameters:
        f (function): a function from R^n to R^n (assume n=1 until Problem 5).
        x0 (float or ndarray): The initial guess for the zero of f.
        Df (function): The derivative of f, a function from R^n to R^(nxn).
        tol (float): Convergence tolerance. The function should returns when
            the difference between successive approximations is less than tol.
        maxiter (int): The maximum number of iterations to compute.
        alpha (float): Backtracking scalar (Problem 3).

    Returns:
        (float or ndarray): The approximation for a zero of f.
        (bool): Whether or not Newton's method converged.
        (int): The number of iterations computed.
    """
    #convergent boolean operator
    converge = False
    #initial point
    x_1 = x0
    #number of iterations
    i = 0

    # if x0 is 1 dimensional the perform univariate Newton's method
    if np.isscalar(x0):
        #repeat until i is maxiter
        while i < maxiter:
            #increment i and find next sequence point
            i += 1
            x_2 = x_1 - (alpha * f(x_1) / Df(x_1))
            #check for convergence
            if abs(x_2 - x_1) < tol:
                #set converge and break
                converge = True
                break
            #reassign x_1 and iterate i
            x_1 = x_2

        return x_2, converge, i

    #perform multivariate Newton's method
    else:
        #repeat until i is maxiter
        while i < maxiter:
            #increment i and find next sequence point
            i += 1
            x_2 = x_1 - (alpha * la.solve(Df(x_1), f(x_1)))
            #check for convergence
            if la.norm(x_2 - x_1) < tol:
                #set converge and break
                converge = True
                break
            #reassign x_1 and iterate i
            x_1 = x_2

        return x_2, converge, i


# Problem 2
def prob2(N1, N2, P1, P2):
    """Use Newton's method to solve for the constant r that satisfies

                P1[(1+r)**N1 - 1] = P2[1 - (1+r)**(-N2)].

    Use r_0 = 0.1 for the initial guess.

    Parameters:
        P1 (float): Amount of money deposited into account at the beginning of
            years 1, 2, ..., N1.
        P2 (float): Amount of money withdrawn at the beginning of years N1+1,
            N1+2, ..., N1+N2.
        N1 (int): Number of years money is deposited.
        N2 (int): Number of years money is withdrawn.

    Returns:
        (float): the value of r that satisfies the equation.
    """

    #initialize our function and derivative to run Newton's method on
    f_r = lambda r: -P1*((1+ r) **N1 - 1) + P2*(1 - (1 + r)**(-N2))
    Df_r = lambda r: -N1*P1*(r + 1)**(N1 - 1) + N2*P2*(r + 1)**(-N2 - 1)
    #get r
    r = nwton(func=f_r, x0=0.1, fprime=Df_r)
    return r



# Problem 4
def optimal_alpha(f, x0, Df, tol=1e-5, maxiter=15):
    """Run Newton's method for various values of alpha in (0,1].
    Plot the alpha value against the number of iterations until convergence.

    Parameters:
        f (function): a function from R^n to R^n (assume n=1 until Problem 5).
        x0 (float or ndarray): The initial guess for the zero of f.
        Df (function): The derivative of f, a function from R^n to R^(nxn).
        tol (float): Convergence tolerance. The function should returns when
            the difference between successive approximations is less than tol.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): a value for alpha that results in the lowest number of
            iterations.
    """

    #get alphas
    alphas = np.linspace(0.1, 1, 10000)
    #convergence boolean
    convergence = False
    #number of iterations for given alpha
    iterations = []
    for alpha in alphas:
        #get convergence value, boolean and iteration number
        val, convergence, iteration = newton(f, x0, Df, 1e-5, 15, alpha)
        #append it
        iterations.append(iteration)

    iterations = np.array(iterations)

    #plot alphas against iterations
    plt.plot(iterations, alphas, 'co-')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Alpha Values')
    plt.show()

    #get the index of the smallest argument
    index = np.argmin(iterations)

    #use it to return the optimal alpha
    return alphas[index]


# Problem 6
def prob6():
    """Consider the following Bioremediation system.

                              5xy − x(1 + y) = 0
                        −xy + (1 − y)(1 + y) = 0

    Find an initial point such that Newton’s method converges to either
    (0,1) or (0,−1) with alpha = 1, and to (3.75, .25) with alpha = 0.55.
    Return the intial point as a 1-D NumPy array with 2 entries.
    """
    #function
    fx = lambda x: np.array([5*x[0]*x[1] - x[0]*(1 + x[1]), -x[0]*x[1] + (1 - x[1])*(1 + x[1])])
    #the jacobian
    Df = lambda x: np.array([[4*x[1] - 1, 4*x[0]],
                             [-x[1], -x[0] - 2*x[1]]])

    #values to range over
    x_range = np.linspace(-1/4., 0, 75)
    y_range = np.linspace(0, 1/4., 75)
    X, Y = np.meshgrid(x_range, y_range)

    #zero points
    point_1 = np.array([0, 1])
    point_2 = np.array([0, -1])
    point_3 = np.array([3.75, 0.25])

    #iterate through x, y pairs as initial point
    for i, row in enumerate(X):
        for j, el in enumerate(row):
            #set initial point
            x0 = np.array([el, Y[i, j]])
            #try catch block to catch singular matrices errors
            try:
                #get computed values
                val_1, convergence_1, _ = newton(fx, x0, Df, alpha = 1)
                val_55, convergence_55, _ = newton(fx, x0, Df, alpha= 0.55)
                #check closeness and return corresponding point
                if np.allclose(val_1, point_1) and np.allclose(val_55, point_3):
                    return x0
                if np.allclose(val_1, point_2) and np.allclose(val_55, point_3):
                    return x0

            except la.LinAlgError as err:
                #catches singular matrix error
                if 'Singular matrix' in str(err):
                    continue
                #if error is not singular matrix raise a NotImplemented Error
                else:
                    raise NotImplementedError

    return None

# Problem 7
def plot_basins(f, Df, zeros, domain, res=1000, iters=15):
    """Plot the basins of attraction of f on the complex plane.

    Parameters:
        f (function): A function from C to C.
        Df (function): The derivative of f, a function from C to C.
        zeros (ndarray): A 1-D array of the zeros of f.
        domain ([r_min, r_max, i_min, i_max]): A list of scalars that define
            the window limits and grid domain for the plot.
        res (int): A scalar that determines the resolution of the plot.
            The visualized grid has shape (res, res).
        iters (int): The exact number of times to iterate Newton's method.
    """
    #create mesh grid and initial points to check
    x_real = np.linspace(domain[0], domain[1], res)
    x_imag = np.linspace(domain[2], domain[3], res)
    X_real, X_imag = np.meshgrid(x_real, x_imag)

    #use to get color values
    _, Y = np.meshgrid(x_real, x_imag)
    #use to calculate newton's method
    X_1 = X_real + 1j*X_imag

    #perform newton's method
    for _ in range(iters):
        X_2 = X_1 - f(X_1) / Df(X_1)
        X_1 = X_2

    #assign index values to Y
    for i in range(res):
        for j in range(res):
            index = np.argmin(np.abs(zeros - X_2[i, j]))
            Y[i, j] = index

    #plot
    plt.pcolormesh(X_real, X_imag, Y, cmap='brg')
    plt.show()




if __name__ == "__main__":

    #problem 1:
    '''
    fx = lambda x: x**4 - 3
    x0 = 1
    Df = lambda x: 4*x**3
    zero, converge, iterations = newton(fx, x0, Df, tol=1e-10)
    print('approximate zero: ', zero)
    print('Plugging in approximate: ', fx(zero))
    print('convergence: ', converge)
    print('number of iterations: ', iterations)
    '''

    #problem 2:
    #TODO: Ask about what function to use
    '''
    N1 = 30
    N2 = 20
    P1 = 2000
    P2 = 8000
    estimated = prob2(N1, N2, P1, P2)
    print(estimated)
    '''

    #problem 3:
    '''
    fx = lambda x: np.sign(x) * np.power(np.abs(x), 1./3)
    Df = lambda x: (1 / 3) * np.power(np.abs(x), -2/3.)
    x0 = 0.01
    val, converge, iterations = newton(fx, x0, Df, tol=1e-5, maxiter=15, alpha=1.)
    val_4, converge_4, iterations_4 = newton(fx, x0, Df, tol=1e-5, maxiter=15, alpha=0.4)
    print('alpha: ', 1)
    print('value: ', val)
    print('Convergence: ', converge)
    print('Iterations: ', iterations)
    print()
    print('alpha: ', 0.4)
    print('value: ', val_4)
    print('Convergence: ', converge_4)
    print('Iterations: ', iterations_4)
    '''

    #problem 4:
    #TODO: Ask about domain
    '''
    fx = lambda x: np.sign(x) * np.power(np.abs(x), 1./3)
    Df = lambda x: (1 / 3) * np.power(np.abs(x), -2/3.)
    x0 = 0.01
    best = optimal_alpha(fx, x0, Df)
    print(best)
    '''

    #problem 5:
    '''
    #TODO ask about anp.Jacobian
    mult = lambda x: np.array([np.cos(x[1]) + np.sin(x[0]) , np.cos(x[0]) + np.sin(x[1])])
    J = lambda x: np.array([[np.cos(x[0]), -np.sin(x[1])],
                            [-np.sin(x[0]), np.cos(x[1])]])

    x0 = np.array([2, 1/3.])
    sol  = newton(mult, x0, J, tol=1e-5, maxiter=15, alpha=1.)
    best = optimal_alpha(mult, x0, J)
    print('Solution: ',  sol[0])
    print('Plugging it in: ', mult(sol[0]))
    print('Best Alpha: ', best)
    '''

    #problem 6:
    '''
    vec = prob6()
    #function
    fx = lambda x: np.array([5*x[0]*x[1] - x[0]*(1 + x[1]), -x[0]*x[1] + (1 - x[1])*(1 + x[1])])
    #the jacobian
    Df = lambda x: np.array([[4*x[1] - 1, 4*x[0]],
                             [-x[1], -x[0] - 2*x[1]]])
    val_1, convergence_1, _ = newton(fx, vec, Df, alpha = 1)
    val_55, convergence_55, _ = newton(fx, vec, Df, maxiter=20, alpha= 0.55)

    print('Alpha: ', 1)
    print('Zero: ', val_1)
    print('Convergence: ', convergence_1)
    print()
    print('Alpha: ', 0.55)
    print('Zero: ', val_55)
    print('Convergence: ', convergence_55)
    '''

    #problem 7:
    #test 1:
    '''
    f = lambda x: x**3 - 1
    Df = lambda x: 3*x**2
    zeros = np.array([1, -1/2. + 1j*np.sqrt(3)/ 2, -1/2. - 1j*np.sqrt(3)/ 2])
    domain = np.array([-1.5, 1.5, -1.5, 1.5])
    plot_basins(f, Df, zeros, domain)
    '''
    #test 2:
    '''
    f = lambda x: x**3 - x
    Df = lambda x: 3*x**2 - 1
    zeros = np.array([-1, 0, 1])
    domain = np.array([-1.5, 1.5, -1.5, 1.5])
    plot_basins(f, Df, zeros, domain)
    '''

