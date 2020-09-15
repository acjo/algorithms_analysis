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
    """Returns the minimum, maximum, and average of the entries of L
    (in that order).
    """
    return min(L), max(L), sum(L) / len( L)

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

# Problem 3
"""Imports the calculator module"""
def hypot(a, b):
    """Calculate and return the length of the hypotenuse of a right triangle.
    all functions imported from the calculator.py module
    """
    return c.sqrt(c.addition(c.product(a,a), c.product(b,b)))

# Problem 4
def power_set(A):
    '''This function outputs a list that mimimics the power set of A'''
    #size of powerset |p(A)|= 2^n where n is the cardinality of A.
    power_list = []
    for i in range(len(A)):
        if i == 0: #adding our empty set
            power_list.append(set())
        else: #getting all subsets of A given of size i
            sub_list = list((combinations(A,i)))
            #I'm adding this nested for loop so that each subset of A will be it's own set
            # as opposed to the entire combination list being it's own set
            for j in range(len(sub_list)): #this iterates over the sub list and makes each
                power_list.append(set(sub_list[j])) # element a set
    power_list.append(A)# adding the set A to our power set
    return power_list #return the power set

if __name__ == "__main__":
    #making the program such that this function does not get imported if this file is used as a module.
    # Problem 5: Implement shut the box.
    def shut_the_box(player, timelimit):
        """Play a single game of shut the box."""
        remaining_numbers = list(range(1,10)) #initialize our set of numbers to remove 1 through 9.
        length = len(remaining_numbers) #gets the length of the remaining nubers list
        start_time = time.time() #starts our timer
        end_time = 0 #initalizes the end time to zero for the condition of the while loop
        while len(remaining_numbers) > 0 and end_time - start_time <= int(timelimit): #executes while there are remaining numbers or while the time has not run out. 
            print("Numbers left: " + str(remaining_numbers)) #prints the remaining numbers
            if sum(remaining_numbers) <= 6: #if the sum of the remaining numbers is less than 6 then the roll has to be between 1 and 6. 
                roll = randint(1,6)
                print("Roll: " + str(roll)) #print the roll
            else:#otherwise the roll is between 1 and 12
                roll = randint(1, 12)
                print("Roll: " + str(roll)) #print the role
            if not box.isvalid(roll, remaining_numbers): #check if the roll is valid, there are numbers left that can sum to the roll
                print("Game over! \n")
                end_time = time.time() #if this isn't the case game is over, set end time variable
                break #exit the while loop/game
            if len(remaining_numbers) == length: #checks if it is the first iteration and just prints the time limit
                print("Seconds left: " + str(timelimit))
            else: #if it isn't the first iteration printing the time limit minus the time played
                print("Seconds left: " + str(round(int(timelimit) - end_time + start_time, 2)))
            player_numbers = [] # intialize a empty lis to hold the player number
            while len(player_numbers) == 0: #enters this while loop if the player numbers length is 0
                player_choice = input("Numbers to eliminate: ") #asks the input number
                player_numbers = box.parse_input(player_choice, remaining_numbers) #parses the input
                if sum(player_numbers) != roll: #makes sure the choice actually sums to the roll
                    print("Invalid inputs \n") #if it doesn't make them choose new numbers and output the time they have left. 
                    end_time = time.time()
                    print("Seconds left: " + str(round(int(timelimit) - end_time + start_time, 2))) #print the time
                    player_numbers = [] #empty the player_numbers list
            for num in player_numbers: #if the numbers are valid remove them from the list
                remaining_numbers.remove(num)
            end_time = time.time() #set end time variable
            print("\n")
        #outputs the score depending on if you won or lost.
        print("Score for player " + str(player) + ": " + str(sum(remaining_numbers)))
        print("Time played: " + str(end_time - start_time))
        if sum(remaining_numbers) == 0 and end_time - start_time <= timelimit:
            print("Congratulations!! You shut the box!")
        else:
            print("Better luck next time! >:)")
        return
#this checks to make sure that the shut the box function can only be called with 3 arguments. otherwise, it won't play the game
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

