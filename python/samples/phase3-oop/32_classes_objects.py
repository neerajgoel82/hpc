"""
Exercise 32: Classes and Objects

Learning Objectives:
- Define a class
- Create objects (instances)
- Instance variables
- Instance methods
- The self parameter
"""

print("=== CREATING A CLASS ===\n")

# Define a simple class
class Dog:
    """A simple class representing a dog"""
    pass

# Create an object (instance)
my_dog = Dog()
print(f"Created dog object: {my_dog}")
print(f"Type: {type(my_dog)}")

# TODO: Create your own simple class


print("\n=== INSTANCE VARIABLES ===\n")

class Car:
    """A class representing a car"""

    def __init__(self, brand, model, year):
        """Initialize car attributes"""
        self.brand = brand  # Instance variable
        self.model = model
        self.year = year

# Create car objects
car1 = Car("Toyota", "Camry", 2020)
car2 = Car("Honda", "Civic", 2021)

# Access attributes
print(f"Car 1: {car1.brand} {car1.model} ({car1.year})")
print(f"Car 2: {car2.brand} {car2.model} ({car2.year})")

# Modify attributes
car1.year = 2021
print(f"Updated car 1: {car1.year}")

# TODO: Create Person class with name, age, city


print("\n=== INSTANCE METHODS ===\n")

class Dog:
    """A class representing a dog"""

    def __init__(self, name, age):
        """Initialize dog attributes"""
        self.name = name
        self.age = age

    def bark(self):
        """Make the dog bark"""
        return f"{self.name} says Woof!"

    def get_age_in_dog_years(self):
        """Calculate age in dog years"""
        return self.age * 7

    def have_birthday(self):
        """Celebrate birthday (increment age)"""
        self.age += 1
        return f"Happy birthday, {self.name}! Now {self.age} years old."

# Create and use dog
my_dog = Dog("Buddy", 3)
print(my_dog.bark())
print(f"Dog years: {my_dog.get_age_in_dog_years()}")
print(my_dog.have_birthday())
print(f"Updated age: {my_dog.age}")

# TODO: Create Rectangle class with width, height, and area() method


print("\n=== THE SELF PARAMETER ===\n")

class Counter:
    """A simple counter class"""

    def __init__(self):
        self.count = 0  # self refers to the instance

    def increment(self):
        self.count += 1  # Access instance variable with self

    def get_count(self):
        return self.count

counter1 = Counter()
counter2 = Counter()

counter1.increment()
counter1.increment()
counter2.increment()

print(f"Counter 1: {counter1.get_count()}")  # 2
print(f"Counter 2: {counter2.get_count()}")  # 1

# Each instance has its own data!

# TODO: Create BankAccount class with deposit and withdraw methods


print("\n=== MULTIPLE INSTANCES ===\n")

class Student:
    """Represent a student"""

    def __init__(self, name, grade):
        self.name = name
        self.grade = grade

    def study(self, hours):
        return f"{self.name} studied for {hours} hours"

    def get_info(self):
        return f"{self.name}: Grade {self.grade}"

# Create multiple students
students = [
    Student("Alice", "A"),
    Student("Bob", "B"),
    Student("Charlie", "A")
]

for student in students:
    print(student.get_info())

# TODO: Create list of Car objects and display them


print("\n=== PRACTICE EXERCISES ===\n")

# Exercise 1: Book Class
# TODO: Create Book class with title, author, pages
# Methods: get_info(), is_long() (>300 pages)


# Exercise 2: Circle Class
# TODO: Create Circle class with radius
# Methods: area(), circumference(), diameter()


# Exercise 3: Temperature Class
# TODO: Create Temperature class with celsius
# Methods: to_fahrenheit(), to_kelvin()


# Exercise 4: TodoItem Class
# TODO: Create TodoItem with description and completed status
# Methods: mark_complete(), is_complete()


# Exercise 5: Timer Class
# TODO: Create Timer with seconds
# Methods: start(), stop(), reset(), get_time()


print("\n=== BONUS CHALLENGES ===\n")

# Challenge 1: Shopping Cart
# TODO: Create ShoppingCart class
# Methods: add_item(), remove_item(), get_total(), clear()


# Challenge 2: Playlist
# TODO: Create Playlist class with songs list
# Methods: add_song(), remove_song(), shuffle(), get_duration()


print("\nExcellent! Next: 33_attributes_methods.py")
