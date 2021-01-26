# differentiation.py
"""Volume 1: Differentiation.
Caelan Osman
Math 347 Sec. 2
Jan. 25th, 2021
"""

import sympy as sy
import numpy as np
from matplotlib import pyplot as plt
from autograd import numpy as anp
from autograd import grad
from autograd import elementwise_grad
import time


# Problem 1
def prob1():
    """Return the derivative of (sin(x) + 1)^sin(cos(x)) using SymPy."""
    #set symbols
    x = sy.symbols('x')
    #return derivative
    return sy.lambdify(x, sy.diff((sy.sin(x) + 1) ** (sy.sin(sy.cos(x)))), 'numpy')


# Problem 2
def fdq1(f, x, h=1e-5):
    """Calculate the first order forward difference quotient of f at x."""
    #calculate and return (array broadcasting)
    return (f(x + h) -f(x)) / h

def fdq2(f, x, h=1e-5):
    """Calculate the second order forward difference quotient of f at x."""
    #calculate and return (array broadcasting)
    return (-3*f(x) + 4*f(x + h) - f(x + 2*h)) / 2*h

def bdq1(f, x, h=1e-5):
    """Calculate the first order backward difference quotient of f at x."""
    #calculate and return (array broadcasting)
    return (f(x) - f(x - h)) / h

def bdq2(f, x, h=1e-5):
    """Calculate the second order backward difference quotient of f at x."""
    #calculate and return (array broadcasting)
    return (3*f(x) - 4*f(x - h) + f(x - 2*h)) / 2*h

def cdq2(f, x, h=1e-5):
    """Calculate the second order centered difference quotient of f at x."""
    #calculate and return (array broadcasting)
    return (f(x + h) - f(x - h)) / 2*h

def cdq4(f, x, h=1e-5):
    """Calculate the fourth order centered difference quotient of f at x."""
    #calculate and return (array broadcasting)
    return (f(x - 2*h) - 8*f(x - h) + 8*f(x+h) - f(x + 2*h)) / 12*h


# Problem 3
def prob3(x0):
    """Let f(x) = (sin(x) + 1)^(sin(cos(x))). Use prob1() to calculate the
    exact value of f'(x0). Then use fdq1(), fdq2(), bdq1(), bdq2(), cdq1(),
    and cdq2() to approximate f'(x0) for h=10^-8, 10^-7, ..., 10^-1, 1.
    Track the absolute error for each trial, then plot the absolute error
    against h on a log-log scale.

    Parameters:
        x0 (float): The point where the derivative is being approximated.
    """
    #set symbols
    x = sy.symbols('x')
    #function
    fx = lambda x: (np.sin(x) + 1)**(np.sin(np.cos(x)))
    #get exact derivative
    deriv = sy.lambdify(x, prob1(), 'numpy')
    #evaluate exact derivative at a point
    exact = deriv(x0)

    #get h values
    hs = np.logspace(-7, 0, 8)

    #get arrays of difference between approx and exact values
    fd1 = np.array([abs(exact - fdq1(fx, x0, h)) for h in hs])
    fd2 = np.array([abs(exact - fdq2(fx, x0, h)) for h in hs])
    bd1 = np.array([abs(exact - bdq1(fx, x0, h)) for h in hs])
    bd2 = np.array([abs(exact - bdq2(fx, x0, h)) for h in hs])
    cd2 = np.array([abs(exact - cdq2(fx, x0, h)) for h in hs])
    cd4 = np.array([abs(exact - cdq4(fx, x0, h)) for h in hs])

    #plot
    plt.loglog(hs, fd1, 'bo-', label = 'Order 1 Forward')
    plt.loglog(hs, fd2, 'o-', color='orange', label = 'Order 2 Forward')
    plt.loglog(hs, bd1, 'go-', label = 'Order 1 Backward')
    plt.loglog(hs, bd2, 'ro-', label = 'Order 2 Backward')
    plt.loglog(hs, cd2, 'o-',color='purple', label = 'Order 2 Centered')
    plt.loglog(hs, cd4, 'o-',color='brown', label = 'Order 4 Centered')
    plt.legend(loc='best')
    plt.xlabel('h')
    plt.ylabel('Absolute Error')
    plt.axis([10e-8, 10e0, 10e-12, 10e0])
    plt.show()





# Problem 4
def prob4():
    """The radar stations A and B, separated by the distance 500m, track a
    plane C by recording the angles alpha and beta at one-second intervals.
    Your goal, back at air traffic control, is to determine the speed of the
    plane.

    Successive readings for alpha and beta at integer times t=7,8,...,14
    are stored in the file plane.npy. Each row in the array represents a
    different reading; the columns are the observation time t, the angle
    alpha (in degrees), and the angle beta (also in degrees), in that order.
    The Cartesian coordinates of the plane can be calculated from the angles
    alpha and beta as follows.

    x(alpha, beta) = a tan(beta) / (tan(beta) - tan(alpha))
    y(alpha, beta) = (a tan(beta) tan(alpha)) / (tan(beta) - tan(alpha))

    Load the data, convert alpha and beta to radians, then compute the
    coordinates x(t) and y(t) at each given t. Approximate x'(t) and y'(t)
    using a first order forward difference quotient for t=7, a first order
    backward difference quotient for t=14, and a second order centered
    difference quotient for t=8,9,...,13. Return the values of the speed at
    each t.
    """
    #get time, alpha, beta arrays
    time, alpha, beta = np.load('plane.npy')[:, 0], np.load('plane.npy')[:, 1], np.load('plane.npy')[:, 2]

    #convert alpha and beta to radians
    alpha = np.radians(alpha)
    beta = np.radians(beta)


    #get the cartesian coordinates of x and y
    xcart = lambda a, b: (a * np.tan(b)) / (np.tan(b) - np.tan(a))
    ycart = lambda a, b: (a * np.tan(b)*np.tan(a)) / (np.tan(b) - np.tan(a))
    xpos = xcart(alpha, beta)
    ypos = ycart(alpha, beta)

    #initalize speed and calculate the corresponding speed
    speed = np.zeros(8)
    for i, t in enumerate(time):
        #use first order forward difference
        if t == 7:
            xprime = xpos[i+1] - xpos[i]
            yprime = ypos[i+1] - ypos[i]
            speed[i] = np.sqrt(xprime**2 + yprime**2)
        #use first order backward difference
        elif t == 14:
            xprime = xpos[i] - xpos[i-1]
            yprime = ypos[i] - ypos[i-1]
            speed[i] = np.sqrt(xprime**2 + yprime**2)
        #use second order centered difference
        else:
            xprime = (xpos[i+1] - xpos[i-1]) / 2
            yprime = (ypos[i+1] - ypos[i-1]) / 2
            speed[i] = np.sqrt(xprime**2 + yprime**2)


    return speed



# Problem 5
def jacobian_cdq2(f, x, h=1e-5):
    """Approximate the Jacobian matrix of f:R^n->R^m at x using the second
    order centered difference quotient.

    Parameters:
        f (function): the multidimensional function to differentiate.
            Accepts a NumPy (n,) ndarray and returns an (m,) ndarray.
            For example, f(x,y) = [x+y, xy**2] could be implemented as follows.
            >>> f = lambda x: np.array([x[0] + x[1], x[0] * x[1]**2])
        x ((n,) ndarray): the point in R^n at which to compute the Jacobian.
        h (float): the step size in the finite difference quotient.

    Returns:
        ((m,n) ndarray) the Jacobian matrix of f at x.
    """
    #get size of of the input
    n = x.size
    #create identity matrix
    I = np.eye(n)
    #calculate Jacobian
    print('Multiplying by h dividng by 2')
    J = np.array([(f(x + h*I[:,j]) -f(x - h*I[:,j])) / h for j in range(n)])

    return J.T


# Problem 6
def cheb_poly(x, n):
    """Compute the nth Chebyshev polynomial at x.

    Parameters:
        x (autograd.ndarray): the points to evaluate T_n(x) at.
        n (int): The degree of the polynomial.
    """
    if n == 0:
        return anp.ones_like(x)
    if n == 1:
        return x

    Tn_2 = anp.ones_like(x)
    Tn_1 = x

    for _ in range(n-1):
        Tn = 2 * x * Tn_1 - Tn_2
        Tn_2 = Tn_1
        Tn_1 = Tn

    return Tn

def prob6():
    """Use Autograd and cheb_poly() to create a function for the derivative
    of the Chebyshev polynomials, and use that function to plot the derivatives
    over the domain [-1,1] for n=0,1,2,3,4.
    """
    domain = anp.linspace(-1, 1, 200)
    ns = [i for i in range(5)]

    cheb_prime = elementwise_grad(cheb_poly)
    for n in ns:
        plt.plot(domain, cheb_prime(domain, n), label='n = ' + str(n))

    plt.axis([-1, 1, -5, 5])
    plt.legend(loc='best')
    plt.title('Chebyshev Derivatives')
    plt.show()


# Problem 7
def prob7(N=200):
    """Let f(x) = (sin(x) + 1)^sin(cos(x)). Perform the following experiment N
    times:

        1. Choose a random value x0.
        2. Use prob1() to calculate the “exact” value of f′(x0). Time how long
            the entire process takes, including calling prob1() (each
            iteration).
        3. Time how long it takes to get an approximation of f'(x0) using
            cdq4(). Record the absolute error of the approximation.
        4. Time how long it takes to get an approximation of f'(x0) using
            Autograd (calling grad() every time). Record the absolute error of
            the approximation.

    Plot the computation times versus the absolute errors on a log-log plot
    with different colors for SymPy, the difference quotient, and Autograd.
    For SymPy, assume an absolute error of 1e-18.
    """

    #initializing empty arrays that will contain the times and absolute error of each function
    problem1_t = []
    problem1_val = []
    center_4t = []
    center_4val = []
    auto_t = []
    auto_val = []
    #initalizing the function and h
    fx = lambda x: (anp.sin(x) + 1)**(anp.sin(anp.cos(x)))
    h = 1e-18

    for _ in range(N):
        x = np.random.randn(1)[0]
        exact = prob1()(x)

        #time problem 1
        sym_start = time.time()
        val1 = prob1()(x)
        sym_end = time.time()
        problem1_t.append(sym_end - sym_start)
        problem1_val.append(abs(exact-val1))

        #time 4th order centered difference quotient
        center_start = time.time()
        val2 = cdq4(fx, x, h)
        center_end = time.time()
        center_4t.append(center_end - center_start)
        center_4val.append(abs(exact - val2))

        #time grad()
        grad_start = time.time()
        val3 = grad(fx)(x)
        grad_end = time.time()
        auto_t.append(grad_end - grad_start)
        auto_val.append(abs(exact - val3))

    plt.plot(problem1_t, problem1_val, 'o', alpha=0.25, label='Sympy')
    plt.plot(center_4t, center_4val, 'o', alpha=0.25, label='Difference')
    plt.plot(auto_t, auto_val, 'o', alpha=0.25, label='Autograd')
    plt.xlabel('Computation Time')
    plt.ylabel('Absolute Error')
    plt.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    #prob1
    '''
    x = sy.symbols('x')
    fx = lambda x: (np.sin(x) + 1)**(np.sin(np.cos(x)))
    deriv = sy.lambdify(x, prob1(), 'numpy')
    domain = np.linspace(-np.pi, np.pi, 1000)
    plt.plot(domain, fx(domain), label = 'f(x)')
    plt.plot(domain, deriv(domain), label="f'(x)")
    plt.legend(loc='best')
    ax = plt.gca()
    ax.spines["bottom"].set_position('zero')
    plt.show()
    '''
    #prob2
    #FIXME: Prob2(), prob3()
    #FIXME: Prob2(), prob3()
    #FIXME: Prob2(), prob3()
    #FIXME: Prob2(), prob3()
    '''
    x = sy.symbols('x')
    fx = lambda x: (np.sin(x) + 1)**(np.sin(np.cos(x)))
    deriv = sy.lambdify(x, prob1(), 'numpy')
    domain = np.linspace(-np.pi, np.pi, 1000)
    #fig, axs = plt.subplots(2, 4)

    outputs = []
    outputs.append(fx(domain))
    outputs.append(deriv(domain))
    outputs.append(fdq1(fx, domain))
    outputs.append(fdq2(fx, domain))
    outputs.append(bdq1(fx, domain))
    outputs.append(bdq2(fx, domain))
    outputs.append(cdq2(fx, domain))
    outputs.append(cdq4(fx, domain))
    labels = ['func', 'deriv', 'fdq1', 'fdq2', 'bdq1', 'bdq2', 'cdq2', 'cdq4']

    func = 0
    width = 6
    for i in range(2):
        for j in range(4):
            #ax = axs[i, j]
            plt.plot(domain, outputs[func], linewidth =width,  label=labels[func])
            width -= 0.5
            #ax.legend(loc='best')
            func += 1

    plt.legend(loc='best')
    plt.show()
    h = 10e-8
    print((fx(1 - 2*h) - 8*fx(1-h) + 8*fx(1+h) - fx(1+2*h)) / 12*h)
    '''
    #prob3
    #FIXME
    #FIXME
    #FIXME
    #FIXME
    #prob3(1)

    #prob4
    #print(prob4())

    #prob5
    #FIXME
    #FIXME
    #FIXME
    #FIXME
    '''
    f = lambda x: np.array([x[0]**2, x[0]**3 - x[1]])
    x = np.array([3, 1])
    print(jacobian_cdq2(f, x))
    '''

    #prob6
    #prob6()


    #prob7
    #FIXME
    #FIXME
    #FIXME
    #FIXME
    #prob7()







