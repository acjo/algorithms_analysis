# standard_library.py
"""Python Essentials: The Standard Library.
Caelan Osman
Math 345 Section 3
September 7, 2020
"""
import calculator as c
"""Used for problem 3"""
from itertools import combinations
"""Used for porblem 4"""
import box
import sys
import time
from random import randint
"""Used for problem 5"""

# Problem 1
def prob1(L):
    """Return the minimum, maximum, and average of the entries of L
    (in that order).
    """
    return min(L), max(L), sum(L) / len( L)
    raise NotImplementedError("Problem 1 Incomplete")

# Problem 2
def prob2():
    """Determine which Python objects are mutable and which are immutable.
    Test numbers, strings, lists, tuples, and sets. Print your results.
    """
    num  = 3
    num_2 = num
    num_2 += 1
    if num_2 == num:
        print("Numbers are mutuable")
    else:
        print("Numbers are immutable")
    string = "ACME is great"
    string_2 = string
    string_2 += "!"
    if string == string_2:
        print("Strings are mutable")
    else:
        print("Strings are immutable")
    myList = [1,2,3]
    myNewList = myList
    myNewList.append(4)
    if myList == myNewList:
        print("Lists are mutuable")
    else:
        print("Lists are immutable")
    myTuple = (1,2,3)
    myNewTuple = myTuple
    myNewTuple += (1,)
    if myTuple == myNewTuple:
        print("Tuples are mutable")
    else:
        print("Tuples are immutable")
    mySet = {"320", "321", "344", "345"}
    myNewSet = mySet
    myNewSet.add("495R")
    if mySet == myNewSet:
        print("Sets are mutable")
    else:
        print("Sets are immutable")
    return
    raise NotImplementedError("Problem 2 Incomplete")

# Problem 3
"""Imports the calculator module"""
def hypot(a, b):
    """Calculate and return the length of the hypotenuse of a right triangle.
    Do not use any functions other than those that are imported from your
    'calculator' module.

    Parameters:
        a: the length one of the sides of the triangle.
        b: the length the other non-hypotenuse side of the triangle.
    Returns:
        The length of the triangle's hypotenuse.
    """
    return c.sqrt(c.addition(c.product(a,a), c.product(b,b)))
    raise NotImplementedError("Problem 3 Incomplete")

# Problem 4
def power_set(A):
    """Use itertools to compute the power set of A.

    Parameters:
        A (iterable): a str, list, set, tuple, or other iterable collection.

    Returns:
        (list(sets)): The power set of A as a list of sets.
    """
    #size of powerset |p(A)|= 2^n where n is the cardinality of A.
    power_list = []
    for i in range(len(A)):
        if i == 0:
            power_list.append(set())
        else:
            sub_list = list((combinations(A,i)))
            #I'm adding this nested for loop so that each subset of A will be it's own set
            # as opposed to the entire combination list being it's own set
            for j in range(len(sub_list)):
                power_list.append(set(sub_list[j]))
    power_list.append(A)
    return power_list
    raise NotImplementedError("Problem 4 Incomplete")
if __name__ == "__main__":
    # Problem 5: Implement shut the box.
    def shut_the_box(player, timelimit):
        """Play a single game of shut the box."""
        remaining_numbers = list(range(1,10))
        length = len(remaining_numbers)
        start_time = time.time()
        end_time = 0
        while len(remaining_numbers) > 0 and end_time - start_time <= int(timelimit):
            print("Numbers left: " + str(remaining_numbers))
            if sum(remaining_numbers) <= 6:
                roll = randint(1,6)
                print("Roll: " + str(roll))
            else:
                roll = randint(1, 12)
                print("Roll: " + str(roll))
            if not box.isvalid(roll, remaining_numbers):
                print("Game over! \n")
                end_time = time.time()
                break
            if len(remaining_numbers) == length:
                print("Seconds left: " + str(timelimit))
            else:
                print("Seconds left: " + str(round(int(timelimit) - end_time + start_time, 2)))
            player_numbers = []
            while len(player_numbers) == 0:
                player_choice = input("Numbers to eliminate: ")
                player_numbers = box.parse_input(player_choice, remaining_numbers)
                if sum(player_numbers) != roll:
                    print("Invalid inputs \n")
                    end_time = time.time()
                    print("Seconds left: " + str(round(int(timelimit) - end_time + start_time, 2)))
                    player_numbers = []
            for num in player_numbers:
                remaining_numbers.remove(num)
            end_time = time.time()
            print("\n")

        print("Score for player " + str(player) + ": " + str(sum(remaining_numbers)))
        print("Time played: " + str(end_time - start_time))
        if sum(remaining_numbers) == 0 and end_time - start_time <= timelimit:
            print("Congratulations!! You shut the box!")
        else:
            print("Better luck next time! >:)")
        return

    if len(sys.argv) == 3:
        shut_the_box(sys.argv[1], sys.argv[2])
    elif len(sys.argv) > 3:
        print("You have " + str(len(sys.arg) - 3) + " too many arguments")
        print("System arguments: ", sys.argv)
    elif len(sys.argv) == 2:
        print("You have exactly one too few arguments")
        print("System arguments: ", sys.argv)
    else:
        print("You have exactly two too few arguments")
        print("System arguments: ", sys.argv)

