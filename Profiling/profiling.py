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
    with open(filename, 'r') as infile:
        data = infile.read().strip().split('\n')

    for i in range(len(data) - 2, -1, -1):
        if i == len(data) - 2:
            previous_l = data[i + 1].split(' ')
        else:
            previous_l = current_l
        current_l = data[i].split(' ')
        for j in range(len(current_l)):
            s1 = int(current_l[j]) + int(previous_l[j])
            s2 = int(current_l[j]) + int(previous_l[j + 1])
            current_l[j] = max([s1, s2])

    return current_l[0]

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

'''
def primes_fast(N):
    """Compute the first N primes."""
    current = 2
    primes_list = []
    l = len(primes_list)
    while l < N:
        isprime = True
        if l > 2:
            string = str(current)
            s = sum([int(num) for num in string])
            if s % 3 == 0:
                current += 2
                continue
        if l > 3:
            if current % 5 == 0:
                current += 2
                continue
        for i in range(3, current, 2):
            #will check for even or divisible and break
            if current % i == 0:
                isprime = False
                #breaking from the for-loop
                break
        if isprime:
            primes_list.append(current)
            l += 1
        if current == 2:
            current += 1
        else:
            current += 2
    return primes_list
'''
def primes_fast(N):
    """Compute the first N primes."""
    current = 2
    primes_list = []
    l = len(primes_list)
    while l < N:
        isprime = True

        if np.any(np.remainder(primes_list, current) == 0):
            current += 2
            continue
        if l > 2:
            if current % 3 == 0:
                current += 2
                continue
        if l > 3:
            if current % 5 == 0:
                current += 2
                continue
        for i in range(3, current, 2):
            #will check for even or divisible and break
            if current % i == 0:
                isprime = False
                #breaking from the for-loop
                break
        if isprime:
            primes_list.append(current)
            l += 1
        if current == 2:
            current += 1
        else:
            current += 2
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
    with open(filename, 'r') as infile:
        names = sorted(infile.read().replace('"', '').split(','))

    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    alphabet_values = {letter: (i + 1) for i, letter in enumerate(alphabet)}

    total = 0
    for i, name in enumerate(names):
        name_value = 0
        for letter in name:
            if letter in alphabet_values:
                name_value += alphabet_values[letter]
        total += name_value * (i + 1)

    return total


# Problem 5
def fibonacci():
    """Yield the terms of the Fibonacci sequence with F_1 = F_2 = 1."""
    raise NotImplementedError("Problem 5 Incomplete")

def fibonacci_digits(N=1000):
    """Return the index of the first term in the Fibonacci sequence with
    N digits.

    Returns:
        (int): The index.
    """
    raise NotImplementedError("Problem 5 Incomplete")


# Problem 6
def prime_sieve(N):
    """Yield all primes that are less than N."""
    raise NotImplementedError("Problem 6 Incomplete")


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

def matrix_power_numba(A, n):
    """Compute A^n, the n-th power of the matrix A, with Numba optimization."""
    raise NotImplementedError("Problem 7 Incomplete")

def prob7(n=10):
    """Time matrix_power(), matrix_power_numba(), and np.linalg.matrix_power()
    on square matrices of increasing size. Plot the times versus the size.
    """
    raise NotImplementedError("Problem 7 Incomplete")



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
    fast = primes_fast(10000)
    end = time.time()
    #print(fast)
    print(end-start)
    #original = primes(100)
    #print(primes(10000)[-1])
    #fast = primes_fast(10)
    #print(fast)
    #print(np.allclose(np.array([original]), np.array([fast])))


    #h =np.any(np.remainder([], 4) == 0)

    '''

    #testing for problem 3
    '''
    A = np.random.randn(200, 100)
    x = np.random.randn(200)
    print(np.all(nearest_column(A, x) == nearest_column_fast(A, x)))
    '''

    #testing for problem 4
    '''
    o = name_scores()
    n = name_scores_fast()
    print(o, n)
    '''





