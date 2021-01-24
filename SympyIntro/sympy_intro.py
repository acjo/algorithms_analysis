# sympy_intro.py
"""Python Essentials: Introduction to SymPy.
Caelan Osman
Math 347 Sec 2
Jan 18, 2021
"""

import sympy as sy
import numpy as np
from matplotlib import pyplot as plt

# Problem 1
def prob1():
    """Return an expression for

        (2/5)e^(x^2 - y)cosh(x+y) + (3/7)log(xy + 1).

    Make sure that the fractions remain symbolic.
    """
    #initialize symbols
    x, y = sy.symbols('x, y')
    #return the expression
    return sy.Rational(2, 5) * sy.exp(x**2 - y)*sy.cosh(x+y) + sy.Rational(3, 7)*sy.log(x*y+ 1)


# Problem 2
def prob2():
    """Compute and simplify the following expression.

        product_(i=1 to 5)[ sum_(j=i to 5)[j(sin(x) + cos(x))] ]
    """
    #initialize symbols
    x, i, j = sy.symbols('x, i, j')
    #create and symblify expression
    return sy.simplify(sy.product(sy.summation(j * (sy.sin(x) + sy.cos(x)), (j, i, 5)), (i, 1, 5)))

# Problem 3
def prob3(N):
    """Define an expression for the Maclaurin series of e^x up to order N.
    Substitute in -y^2 for x to get a truncated Maclaurin series of e^(-y^2).
    Lambdify the resulting expression and plot the series on the domain
    y in [-3,3]. Plot e^(-y^2) over the same domain for comparison.
    """
    #set symbols
    x, n, y = sy.symbols('x, n, y')
    #define the mclaurin series
    exp = sy.summation( x**n / sy.factorial(n), (n, 0, N))
    #sub in -y^2 for x
    eyp = exp.subs(x, -y**2)
    #lambdify the expression and plot with actual
    mclaurin = sy.lambdify(y, eyp, "numpy")
    exact = lambda y: np.exp(-y**2)
    domain = np.linspace(-2, 2, 200)
    plt.plot(domain, exact(domain), 'r', label='Exact')
    plt.plot(domain, mclaurin(domain), 'c', label='McLaurin')
    plt.legend(loc='best')
    plt.title('McLaurin Accuracy')
    plt.show()

    return


# Problem 4
def prob4():
    """The following equation represents a rose curve in cartesian coordinates.

    0 = 1 - [(x^2 + y^2)^(7/2) + 18x^5 y - 60x^3 y^3 + 18x y^5] / (x^2 + y^2)^3

    Construct an expression for the nonzero side of the equation and convert
    it to polar coordinates. Simplify the result, then solve it for r.
    Lambdify a solution and use it to plot x against y for theta in [0, 2pi].
    """
    #set symbols
    x, y, r, t = sy.symbols('x, y, r, t')
    #create the initial expression
    rose = 1 - ((x**2 + y**2)**sy.Rational(7, 2) + 18*x**5*y - 60*x**3*y**3 + 18*x*y**5)/ (x**2 + y**2)**3
    #sub in polar expressions and simplify
    polar = sy.simplify(rose.subs({x:r*sy.cos(t), y:r*sy.sin(t)}))
    #solve and pick the first solution
    rose_expr = sy.solve(polar, r)[0]
    #lambdify the expression
    rose_f = sy.lambdify(t, rose_expr, 'numpy')
    #plot expression
    domain = np.linspace(0, 2*np.pi, 250)
    plt.plot(rose_f(domain) * np.cos(domain), rose_f(domain) * np.sin(domain), 'm')
    plt.title('Rose Curve')
    plt.show()
    return


# Problem 5
def prob5():
    """Calculate the eigenvalues and eigenvectors of the following matrix.

            [x-y,   x,   0]
        A = [  x, x-y,   x]
            [  0,   x, x-y]

    Returns:
        (dict): a dictionary mapping eigenvalues (as expressions) to the
            corresponding eigenvectors (as SymPy matrices).
    """
    #initialize symobls
    x, y, lam = sy.symbols('x, y, lam')

    #define both matrices
    A = sy.Matrix([[x-y, x, 0],
                   [x, x-y, x],
                   [0, x, x-y]])
    L = sy.Matrix([[lam, 0, 0],
                   [0, lam, 0],
                   [0, 0, lam]])

    #get characteristic polynomial
    characteristic_p = (A - L).det()
    #get eigenvalues
    eigen_vals = sy.solve(characteristic_p, lam)

    #create and return dictionary containing eigenvals (key) and eigenvecs (value)
    return {val : (A - L.subs(lam, val)).nullspace() for val in eigen_vals}






# Problem 6
def prob6():
    """Consider the following polynomial.

        p(x) = 2*x^6 - 51*x^4 + 48*x^3 + 312*x^2 - 576*x - 100

    Plot the polynomial and its critical points. Determine which points are
    maxima and which are minima.

    Returns:
        (set): the local minima.
        (set): the local maxima.
    """
    #create the domain
    domain = np.linspace(-5, 5, 100)
    #initialze our symbol
    x = sy.symbols('x')
    #initialize expression and lambda versions of the functio
    expr = 2*x**6 - 51*x**4 + 48*x**3 + 312*x**2 - 576*x - 100
    function = lambda x: 2*x**6 - 51*x**4 + 48*x**3 + 312*x**2 - 576*x - 100
    #take the first and second derivative, lambdify the second derivative
    deriv_1 = sy.diff(expr, x)
    deriv_2 = sy.lambdify(x, sy.diff(deriv_1, x), 'numpy')

    #solve for the critical points
    critical_points = sy.solve(deriv_1, x)

    #initialize local_min, local_max as empty lists
    local_min = []
    local_max = []
    #append all necessary points to the correspoinding list
    for point in critical_points:
        if float(deriv_2(point)) > 0:
            local_min.append(point)
        elif float(deriv_2(point)) < 0:
            local_max.append(point)

    #plot function and min / max values
    plt.plot(domain, function(domain), 'k', label='Function')
    plt.plot(local_min, function(np.array(local_min)), 'co', markersize=6, label='Minima')
    plt.plot(local_max, function(np.array(local_max)), 'bo', markersize=6, label='Maxima')
    plt.legend(loc='best')
    plt.title('Max and Min')
    plt.show()

    #return the local min and local max lists as sets
    return set(local_min), set(local_max)


# Problem 7
def prob7():
    """Calculate the integral of f(x,y,z) = (x^2 + y^2 + z^2)^2 over the
    sphere of radius r. Lambdify the resulting expression and plot the integral
    value for r in [0,3]. Return the value of the integral when r = 2.

    Returns:
        (float): the integral of f over the sphere of radius 2.
    """
    #intialize symbols
    x, y, z, r, p, t = sy.symbols('x, y, z, r, p, t')

    #setting up initial expression
    expr = (x**2 + y**2 + z**2)**2

    #subbing in spherical coordinates
    spherical = expr.subs({x:r*sy.sin(p)*sy.cos(t), y:r*sy.sin(p)*sy.sin(t),z: r*sy.cos(p)})

    #create the Jacobian
    J = sy.Matrix([[r*sy.sin(p)*sy.cos(t)],
                  [r*sy.sin(p)*sy.sin(t)],
                  [r * sy.cos(p)]]).jacobian([r, p, t])

    #multiply spherical expression by the Jacobian
    spherical *= J.det()
    #symplify the spherical equation
    spherical = sy.simplify(spherical)

    #integrate and create a lambda function of r
    integral_r = sy.lambdify(r, sy.integrate(spherical, (p, 0, sy.pi), (t, 0, 2*sy.pi), (r, 0, r)), 'numpy')

    #plot
    r = np.linspace(0, 3, 150)
    plt.plot(r, integral_r(r), 'r-.', label = 'r in [0, 3]' )
    plt.title('Integral values')
    plt.legend(loc='best')
    plt.show()

    return integral_r(2)


if __name__ == "__main__":
    '''
    x, y = sy.symbols('x, y')
    A = sy.Matrix([[x-y, x, 0],
                   [x, x-y, x],
                   [0, x, x-y]])
    valuation = prob5()

    keys = valuation.keys()

    for key in keys:
        vec = valuation[key]
        for v in vec:
            Avec = A @ v
            lamvec = key * v
            break
        print(sy.simplify(Avec))
        print(sy.simplify(lamvec))
    '''
