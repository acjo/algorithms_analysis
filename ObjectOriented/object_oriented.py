# object_oriented.py
"""Python Essentials: Object Oriented Programming.
Caelan Osman
Math 345
September 15, 2020
"""


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

def test_backpack():
    ''' This function tests the backpack class to make sure it is working properly.
    '''
    test_pack = Backpack("Caelan","Green")
    if test_pack.name != "Caelan":
        Print("Backpack name assigned incorrectly!")
    if test_pack.color != "Green":
        Print("Backpack color assigned incorrectly!")
    for item in ["Phone", "laptop", "Notebook", "Pen"]:
        test_pack.put(item)
    print("Contents:", test_pack.contents)
    # Magic Methods -----------------------------------------------------------

    # Problem 3: Write __eq__() and __str__().
    def __add__(self, other):
        """Add the number of contents of each Backpack."""
        return len(self.contents) + len(other.contents)

    def __lt__(self, other):
        """Compare two backpacks. If 'self' has fewer contents
        than 'other', return True. Otherwise, return False.
        """
        return len(self.contents) < len(other.contents)


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

    def __init__(name, color, max_size = 2 , fuel_amount = 10):
        """ Overriding the superclass' constructor so we can add fuel_ammount
        """
        self.name = name
        self.color = color
        self.max_size = 5
        self.fuel_amount = 10
        self.contents =[]

    def fly(burn_fuel):
        '''define a new method so we can "fly" but if the fuel to burn (burn_fuel) is greater than
           the amount of fuel we have then print that we don't have enough fuel otherwise take away the
           fuel from the amount of fuel stored by the class
        '''
        if burn_fuel > self.fuel_amount:
            print("Not enough fuel!")
        else:
            self.fuel_amount = self.fuel_amount - burn_fuel
    def dump():
        """ Override the dump() method so we dump both the contents and set the fuel_amount to 0.
        """
        self.contents.clear()
        self.fuel_amount = 0



# Problem 4: Write a 'ComplexNumber' class.
