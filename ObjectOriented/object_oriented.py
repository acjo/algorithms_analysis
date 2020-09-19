# object_oriented.py
"""Python Essentials: Object Oriented Programming.
Caelan Osman
Math 345
September 15, 2020
"""

from math import sqrt

class Backpack:
    """A Backpack object class. Has a name, color, max size
    and a list of contents.

    Attributes:
        name (str): the name of the backpack's owner.
        contents (list): the contents of the backpack.
        color(str)
        max_size(int)
    """

    def __init__(self, name, color, max_size = 5):
        """Set the name, color, and max_size and initialize an empty list of contents.

        Parameters:
            name (str): the name of the backpack's owner.
            color (str): the color of the backpack
            max_size(int): the maximum number of items the backpack can hold
        """
        self.name = name
        self.contents = []
        self.color = color
        self.max_size = max_size

    def put(self, item):
        '''Adds item to the list of contents as long as the contents isn't too full
        '''
        if len(self.contents) >= self.max_size:
            print("No Room!")
        else:
            self.contents.append(item)

    def dump(self):
        ''' This function dumps all the contents if called.
        '''
        self.contents.clear()

    def take(self, item):
        """Remove an item from the backpack's list of contents."""
        self.contents.remove(item)

    # Magic Methods -----------------------------------------------------------
    # Problem 3: Write __eq__() and __str__().

    def __eq__(self, other):
        """Returns a boolean if the two objects are the same or not
        """
        if self.color == other.color and self.name == other.name and len(self.contents) == len(other.contents):
            return True
        else:
            return False

    def __str__(self):
        """Returns a string that can be printed out that represents an object of the backpack class
        """
        return "Owner:\t\t" + self.name + "\nColor:\t\t" + self.color + "\nSize:\t\t" + str(len(self.contents)) + "\nMax Size:\t" + str(self.max_size)+ "\nContents:\t" + str(self.contents)

    def __add__(self, other):
        """Add the number of contents of each Backpack."""
        return len(self.contents) + len(other.contents)

    def __lt__(self, other):
        """Compare two backpacks. If 'self' has fewer contents
        than 'other', return True. Otherwise, return False.
        """
        return len(self.contents) < len(other.contents)

def test_backpack():
    ''' This function tests the backpack class to make sure it is working properly.
    '''
    test_pack = Backpack("Caelan","Green")
    if test_pack.name != "Caelan":
        print("Backpack name assigned incorrectly!")
    if test_pack.color != "Green":
        print("Backpack color assigned incorrectly!")
    for item in ["Phone", "laptop", "Notebook", "Pen"]:
        test_pack.put(item)
    print("Contents:", test_pack.contents)
    test_pack_2 = Backpack("Caelan", "Green")
    for item in ["Phone", "laptop", "Notebook", "Pencil"]:
        test_pack_2.put(item)
    print(test_pack_2)
    print(test_pack_2 == test_pack)

test_backpack()

class Knapsack(Backpack):
    """A Knapsack object class. Inherits from the Backpack class.
    A knapsack is smaller than a backpack and can be tied closed.

    Attributes:
        name (str): the name of the knapsack's owner.
        color (str): the color of the knapsack.
        max_size (int): the maximum number of items that can fit inside.
        contents (list): the contents of the backpack.
        closed (bool): whether or not the knapsack is tied shut.
    """
    def __init__(self, name, color):
        """Use the Backpack constructor to initialize the name, color,
        and max_size attributes. A knapsack only holds 3 item by default.

        Parameters:
            name (str): the name of the knapsack's owner.
            color (str): the color of the knapsack.
            max_size (int): the maximum number of items that can fit inside.
        """
        Backpack.__init__(self, name, color, max_size=3)
        self.closed = True

    def put(self, item):
        """If the knapsack is untied, use the Backpack.put() method."""
        if self.closed:
            print("I'm closed!")
        else:
            Backpack.put(self, item)

    def take(self, item):
        """If the knapsack is untied, use the Backpack.take() method."""
        if self.closed:
            print("I'm closed!")
        else:
            Backpack.take(self, item)

    def weight(self):
        """Calculate the weight of the knapsack by counting the length of the
        string representations of each item in the contents list.
        """
        return sum(len(str(item)) for item in self.contents)


# Problem 2: Write a 'Jetpack' class that inherits from the 'Backpack' class.

class Jetpack(Backpack):
    """ A Jetpack class objet has a namce, color, max_size, fuel_amount, and a list of contents
        Attributes:
            name (str): the name of the backpack's owner.
            contents (list): the contents of the backpack.
            color(str): The color of the jet pack
            max_size(int): the max size the jetpack can hold
            fuel_ammount(int): the amount of fuel the jetpack holds
    """

    def __init__(self, name, color, max_size = 2 , fuel_amount = 10):
        """ Overriding the superclass' constructor so we can add fuel_ammount
        """
        self.name = name
        self.color = color
        self.max_size = 5
        self.fuel_amount = 10
        self.contents =[]

    def fly(self, burn_fuel):
        '''define a new method so we can "fly" but if the fuel to burn (burn_fuel) is greater than
           the amount of fuel we have then print that we don't have enough fuel otherwise take away the
           fuel from the amount of fuel stored by the class
        '''
        if burn_fuel > self.fuel_amount:
            print("Not enough fuel!")
        else:
            self.fuel_amount = self.fuel_amount - burn_fuel
    def dump(self):
        """ Override the dump() method so we dump both the contents and set the fuel_amount to 0.
        """
        self.contents.clear()
        self.fuel_amount = 0


# Problem 4: Write a 'ComplexNumber' class.
class ComplexNumber:

    def __init__(self,a,b):
        """Constructor for initializing a complex number"""
        self.real = a
        self.imag = b
    def conjugate(self):
        """returns the complex conjuigate as a new ComplexNumber object"""
        return ComplexNumber(self.real, -self.imag)
    def __str__(self):
        """Prints the complex number as (a+bj) or (a-bj)"""
        if (self.imag >= 0):
            return "(" + str(self.real) + "+" + str(self.imag) + "j)"
        else:
            return "(" + str(self.real)  + str(self.imag) + "j)"
    def __abs__(self):
        """Returns the "norm" of the complex number"""
        return sqrt((self.real **2)+ (self.imag **2))
    def __eq__(self, other):
        """returns true or false if two complex numbers have the exact same elements"""
        if self.real == other.real and self.imag == other.imag:
            return True
        else:
            return False
    def __add__(self, other):
        """Adds two complex numbers together and returns a new complex number"""
        return ComplexNumber(self.real + other.real, self.imag + other.imag)
    def __sub__(self, other):
        """Subtracts two complex numbers and returns a new complex number"""
        return ComplexNumber(self.real - other.real, self.imag - other.imag)
    def __mul__(self, other):
        """Multiplies two complex numbers togher using (a+bi)(c+di) = (ac-bd)+(ad+bc)i and returns
           a new complex number
        """
        return ComplexNumber((self.real * other.real) - (self.imag * other.imag), ((self.real * other.imag) +(self.imag * other.real)))
    def __truediv__(self, other):
        """Divides two complex numbers together using ((a+bi)/(c+di)*((c-di)/(c-di)))
           and returns a new complex number.
        """
        return ComplexNumber(((self.real * other.real + self.imag * other.imag)/(other.real**2 + other.imag**2)), ((other.real * self.imag - self.real * other.imag)/ (other.real**2 + other.imag**2)))


def test_ComplexNumber(a, b):
    py_cnum, my_cnum = complex(a, b), ComplexNumber(a, b)
    # Validate the constructor.
    if my_cnum.real != a or my_cnum.imag != b:
        print("__init__() set self.real and self.imag incorrectly")
    # Validate conjugate() by checking the new number's imag attribute.
    if py_cnum.conjugate().imag != my_cnum.conjugate().imag:
        print("conjugate() failed for", py_cnum)
    # Validate __str__().
    if str(py_cnum) != str(my_cnum):
        print("__str__() failed for", py_cnum)
    print(my_cnum)
    print(my_cnum.conjugate())
    conj = my_cnum.conjugate()
    print(abs(my_cnum))
    print(conj == my_cnum)
    print(conj + my_cnum)
    print(conj - my_cnum)
    print(my_cnum * conj)
    print(my_cnum / conj)



test_ComplexNumber(1,4)








