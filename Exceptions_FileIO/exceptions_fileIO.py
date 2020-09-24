# exceptions_fileIO.py
"""Python Essentials: Exceptions and File Input/Output.
Caelan Osman
Math 345 Sec 3
September 24, 2020
<Date>
"""

from random import choice


# Problem 1
def arithmagic():
    """
    Takes in user input to perform a magic trick and prints the result.
    Verifies the user's input at each step and raises a
    ValueError with an informative error message if any of the following occur:
    The first number step_1 is not a 3-digit number.
    The first number's first and last digits differ by less than $2$.
    The second number step_2 is not the reverse of the first number.
    The third number step_3 is not the positive difference of the first two numbers.
    The fourth number step_4 is not the reverse of the third number.
    """
    step_1 = input("Enter a 3-digit number where the first and last "
                                           "digits differ by 2 or more: ")
    if len(step_1) != 3:
        raise ValueError('The first number (' + step_1 +') is not a 3-digit number.')
    elif abs(int(step_1[0]) - int(step_1[2])) < 2:
        raise ValueError('The number\'s first and last digits differ by less than 2.')

    step_2 = input("Enter the reverse of the first number, obtained "
                                              "by reading it backwards: ")
    if step_1[::-1] != step_2:
        raise ValueError('The second number (' + str(step_2) + ') is not the reverse of the first number.')
    step_3 = input("Enter the positive difference of these numbers: ")
    if abs(int(step_1) - int(step_2)) != int(step_3):
        raise ValueError('The third number (' + step_3 + ') is not the positive difference of the first two numbers.')
    step_4 = input("Enter the reverse of the previous result: ")
    if step_3[::-1] != step_4:
        raise ValueError('The fourth number (' + step_4 + ') is not the reverse of the third number.')
    print(str(step_3), "+", str(step_4), "= 1089 (ta-da!)")

# Problem 2
def random_walk(max_iters=1e12):
    """
    If the user raises a KeyboardInterrupt by pressing ctrl+c while the
    program is running, the function should catch the exception and
    print "Process interrupted at iteration $i$".
    If no KeyboardInterrupt is raised, print "Process completed".

    Return walk.
    """

    walk = 0
    directions = [1, -1]
    interrupt = 0
    for i in range(int(max_iters)):
        try:
            walk += choice(directions)
        except KeyboardInterrupt:
            interrupt = 1
            print(" Process interrupted at iteration:", i)
            break
    if interrupt == 0:
        print("Process passed")
    return walk

# Problems 3 and 4: Write a 'ContentFilter' class.
"""Class for reading in file

    Attributes:
        filename (str): The name of the file
        contents (str): the contents of the file
        
    """
class ContentFilter(object):   
    # Problem 3
    def __init__(self, filename):
        """Read from the specified file. If the filename is invalid, prompt
        the user until a valid filename is given.
        """
    
 # Problem 4 ---------------------------------------------------------------
    def check_mode(self, mode):
        """Raise a ValueError if the mode is invalid."""

    def uniform(self, outfile, mode='w', case='upper'):
        """Write the data ot the outfile in uniform case."""


    def reverse(self, outfile, mode='w', unit='word'):
        """Write the data to the outfile in reverse order."""

    def transpose(self, outfile, mode='w'):
        """Write the transposed version of the data to the outfile."""

    def __str__(self):
        """String representation: info about the contents of the file."""
