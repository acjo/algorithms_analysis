# profiling.py
"""Python Essentials: Profiling.
Caelan Osman
Math 347 Section 2
January 10, 2021
"""

# Note: for problems 1-4, you need only implement the second function listed.
# For example, you need to write max_path_fast(), but keep max_path() unchanged
# so you can do a before-and-after comparison.

import numpy as np
import time
from numpy import linalg as la
from numba import jit
from matplotlib import pyplot as plt


# Problem 1
def max_path(filename="triangle.txt"):
    """Find the maximum vertical path in a triangle of values."""
    with open(filename, 'r') as infile:
        data = [[int(n) for n in line.split()]
                        for line in infile.readlines()]
    def path_sum(r, c, total):
        """Recursively compute the max sum of the path starting in row r
        and column c, given the current total.
        """
        total += data[r][c]
        if r == len(data) - 1:          # Base case.
            return total
        else:                           # Recursive case.
            return max(path_sum(r+1, c,   total),   # Next row, same column
                       path_sum(r+1, c+1, total))   # Next row, next column

    return path_sum(0, 0, 0)            # Start the recursion from the top.

def max_path_fast(filename="triangle_large.txt"):
    """Find the maximum vertical path in a triangle of values."""
    #get the data
    with open(filename, 'r') as infile:
        data = infile.read().strip().split('\n')

    #iterate from bottom
    for i in range(len(data) - 2, -1, -1):
        #if we are just starting set previous to the last row
        if i == len(data) - 2:
            prev_line = data[i + 1].split(' ')
        #otherwise set previous_line to current
        else:
            prev_line = curr_line
        #set current line
        curr_line = data[i].split(' ')
        #iterate through the current line and get the max of the current number
        #and the two below it
        for j, number in enumerate(curr_line):
            s1 = int(number) + int(prev_line[j])
            s2 = int(number) + int(prev_line[j + 1])
            curr_line[j] = max([s1, s2])

    #return the first number as the max
    return curr_line[0]

# Problem 2
def primes(N):
    """Compute the first N primes."""
    primes_list = []
    current = 2
    while len(primes_list) < N:
        isprime = True
        for i in range(2, current): # Check for nontrivial divisors.
            if current % i == 0:
                isprime = False
        if isprime:
            primes_list.append(current)
        current += 1
    return primes_list

def primes_fast(N):
    """Compute the first N primes."""
    # return an empty list
    if N == 0:
        return []
    #initalize list with 2
    primes_list = [2]
    #set current to 3
    current = 3
    #initialize length
    length = len(primes_list)
    while length < N:
        isprime = True
        #calculate root of current
        root = int(np.sqrt(current))
        for prime in primes_list:
            #the current prime is greater than the
            #sqrt of the current it has to be prime
            if prime > root:
                break
            #if the current is divisble by prime then it
            #can't be prime
            elif current % prime == 0:
                isprime = False
                break
        if isprime:
            #add prime to list and increment length
            primes_list.append(current)
            length += 1
        #count up by two
        current += 2

    #return list of primes
    return primes_list


# Problem 3
def nearest_column(A, x):
    """Find the index of the column of A that is closest to x.

    Parameters:
        A ((m,n) ndarray)
        x ((m, ) ndarray)

    Returns:
        (int): The index of the column of A that is closest in norm to x.
    """
    distances = []
    for j in range(A.shape[1]):
        distances.append(np.linalg.norm(A[:,j] - x))
    return np.argmin(distances)

def nearest_column_fast(A, x):
    """Find the index of the column of A that is closest in norm to x.
    Refrain from using any loops or list comprehensions.

    Parameters:
        A ((m,n) ndarray)
        x ((m, ) ndarray)

    Returns:
        (int): The index of the column of A that is closest in norm to x.
    """
    #return the index of the column with the smallest norm in A - x
    #have to stack it so array broadcasting works correctl
    return np.argmin(np.linalg.norm(A - np.vstack(x), axis=0))

# Problem 4
def name_scores(filename="names.txt"):
    """Find the total of the name scores in the given file."""
    with open(filename, 'r') as infile:
        names = sorted(infile.read().replace('"', '').split(','))
    total = 0
    for i in range(len(names)):
        name_value = 0
        for j in range(len(names[i])):
            alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            for k in range(len(alphabet)):
                if names[i][j] == alphabet[k]:
                    letter_value = k + 1
            name_value += letter_value
        total += (names.index(names[i]) + 1) * name_value
    return total

def name_scores_fast(filename='names.txt'):
    """Find the total of the name scores in the given file."""
    #import the names
    with open(filename, 'r') as infile:
        names = sorted(infile.read().replace('"', '').split(','))

    #set our alphabet
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    #create a dictionary of key value pairs for each letter in alphabet
    alphabet_values = {letter: (i + 1) for i, letter in enumerate(alphabet)}

    #total count initialize to zero
    total = 0
    #iterate through all names
    for i, name in enumerate(names):
        #initial name value to zero
        name_value = 0
        for letter in name:
            #get the value for the letter key and add it to name value
            if letter in alphabet_values:
                name_value += alphabet_values[letter]
        #multiply name value by the score (index of name + 1) and add to total
        total += name_value * (i + 1)

    return total


# Problem 5
def fibonacci():
    """Yield the terms of the Fibonacci sequence with F_1 = F_2 = 1."""
    #booleans to help with the loop and what to return
    first = True
    second = True
    cont = True
    while cont:
        #if it is not the first and not the second yield Fn = Fn_2 + Fn_1
        #and set Fn_2 = Fn_1 and Fn_1 = Fn
        if not first and not second:
            Fn = Fn_1 + Fn_2
            yield Fn
            Fn_2 = Fn_1
            Fn_1 = Fn
        #if it is first set Fn_2 = 1 and yield Fn_2
        #set first to False
        if first:
            Fn_2 = 1
            yield Fn_2
            first = False
        #if it is first set Fn_1 = 1 and yiled Fn_2
        #set second to false
        if second:
            Fn_1 = 1
            yield Fn_1
            second = False

def fibonacci_digits(N=1000):
    """Return the index of the first term in the Fibonacci sequence with
    N digits.

    Returns:
        (int): The index.
    """
    #get the generator
    fib = fibonacci()
    #initial index and digits to 0
    index = 0
    digits = 0
    while digits < N:
        #increment index
        index += 1
        #call the generator to get current number
        current = next(fib)
        #set digits to the length of the string
        #representation of current
        digits = len(str(current))

    #return the index
    return index

# Problem 6
def prime_sieve(N):
    """Yield all primes that are less than N."""
    # No primes are less than or equal to 1
    if N <= 1:
        return
    #set the numbers sipping by 2 from 3
    numbers = np.array([num for num in range(3, N+1, 2)])
    #yield 2
    yield 2
    while numbers.size > 0:
        #find which elements of numbers is divisible by the first element
        mask = np.remainder(numbers, numbers[0]) == 0
        #yield the first
        yield numbers[0]
        #apply the negation of the mask to numbers
        numbers = numbers[~mask]


# Problem 7
def matrix_power(A, n):
    """Compute A^n, the n-th power of the matrix A."""
    product = A.copy()
    temporary_array = np.empty_like(A[0])
    m = A.shape[0]
    for power in range(1, n):
        for i in range(m):
            for j in range(m):
                total = 0
                for k in range(m):
                    total += product[i,k] * A[k,j]
                temporary_array[j] = total
            product[i] = temporary_array
    return product

#apply the @jit function decorator for numba enhancement
@jit
def matrix_power_numba(A, n):
    """Compute A^n, the n-th power of the matrix A, with Numba optimization."""
    product = A.copy()
    temporary_array = np.empty_like(A[0])
    m = A.shape[0]
    for power in range(1, n):
        for i in range(m):
            for j in range(m):
                total = 0
                for k in range(m):
                    total += product[i, k] * A[k ,j]
                temporary_array[j] = total
            product[i] = temporary_array
    return product

def prob7(n=10):
    """Time matrix_power(), matrix_power_numba(), and np.linalg.matrix_power()
    on square matrices of increasing size. Plot the times versus the size.
    """
    #set A as a random small matrix and call matrix_power_numba() to compile function
    A = np.random.random((2, 2))
    matrix_power_numba(A, 1)

    #initialize empty lists for numba time, numpy time, and python time
    numba = []
    num_py = []
    python = []
    #set our ms
    ms = [num for num in range(2, 8)]
    for m in ms:
        #get A
        A = np.random.random((2**m, 2**m))

        #time numba
        numba_start = time.time()
        matrix_power_numba(A, n)
        numba_end = time.time()
        numba.append(numba_end - numba_start)

        #time numpy function
        numpy_start = time.time()
        la.matrix_power(A, n)
        numpy_end = time.time()
        num_py.append(numpy_end - numpy_start)

        #time python function
        python_start = time.time()
        matrix_power(A, n)
        python_end = time.time()
        python.append(python_end - python_start)

    #plot on a logplot the power (m) with respect to the times
    plt.loglog(ms, numba,'c-', label= 'Numba Optimized')
    plt.loglog(ms, num_py, 'r--', label='NumPy Optimized')
    plt.loglog(ms, python, 'm-.', label='Python Code')
    plt.title('Time Differences Matrix Power')
    plt.legend(loc='best')
    plt.xlabel("Matrix Size")
    plt.ylabel("Time (s)")
    plt.show()







if __name__ ==  "__main__":

    #testing for problem 1
    '''
    start = time.time()
    max_path()
    end = time.time()
    time_1 = end - start
    start = time.time()
    max_path_fast('triangle.txt')
    end = time.time()
    time_2 = end - start
    print(time_1, time_2)
    print(time_1 / time_2)
    '''
    #testing for problem 2
    '''
    start = time.time()
    fast = primes_fast(1000)
    end = time.time()
    slow = primes_fast(1000)
    print(fast == slow)
    print(end-start)
    '''

    #testing for problem 3
    '''
    A = np.random.randn(200, 100)
    x = np.random.randn(200)
    print(nearest_column(A, x) == nearest_column_fast(A, x))
    '''

    #testing for problem 4
    '''
    n = name_scores_fast()
    o = name_scores()
    print(o, n)
    '''

    #testing for problem 5
    '''
    x = fibonacci()
    print(next(x))
    print(next(x))
    print(next(x))
    print(next(x))
    print(next(x))
    print(next(x))
    print(next(x))
    print(next(x))
    print(next(x))
    print(next(x))
    print(next(x))
    print(next(x))
    '''

    #print(fibonacci_digits(3))

    #testing for problem 6
    '''
    N = 100000
    x = prime_sieve(N)
    p = []
    p.append(next(x))

    while len(p) <= 9591:
        p.append(next(x))
    tf = time.time()

    fast = primes_fast(9592)

    print(np.allclose(np.array(fast), np.array(p)))
    '''

    #testing for problem 7
    #prob7(n=10)







