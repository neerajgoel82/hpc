"""
Exercise 16: Functions Basics

Learning Objectives:
- Define and call functions
- Function parameters
- Return values
- Docstrings
- Why use functions
"""

print("=== DEFINING FUNCTIONS ===\n")

# Basic function definition
def greet():
    """Print a greeting message"""
    print("Hello, World!")

# Call the function
greet()
greet()  # Can call multiple times

# Function with parameter
def greet_person(name):
    """Greet a person by name"""
    print(f"Hello, {name}!")

greet_person("Alice")
greet_person("Bob")

# TODO: Create function to print your favorite quote


print("\n=== RETURN VALUES ===\n")

# Function that returns a value
def add(a, b):
    """Add two numbers and return result"""
    return a + b

result = add(5, 3)
print(f"5 + 3 = {result}")

# Use return value directly
print(f"10 + 20 = {add(10, 20)}")

# Function without return (returns None)
def say_hello():
    print("Hello")

result = say_hello()
print(f"Result: {result}")  # None

# TODO: Create function that multiplies two numbers


print("\n=== MULTIPLE PARAMETERS ===\n")

def introduce(name, age, city):
    """Introduce a person"""
    return f"{name} is {age} years old and lives in {city}"

intro = introduce("Alice", 25, "Paris")
print(intro)

# Order matters with positional arguments
print(introduce("Bob", 30, "London"))

# TODO: Create function that takes 3 numbers and returns their average


print("\n=== DEFAULT PARAMETERS ===\n")

def greet_with_title(name, title="Mr."):
    """Greet with optional title"""
    return f"Hello, {title} {name}"

print(greet_with_title("Smith"))  # Uses default
print(greet_with_title("Smith", "Dr."))  # Override default

def power(base, exponent=2):
    """Calculate power (default: square)"""
    return base ** exponent

print(f"3^2 = {power(3)}")
print(f"3^3 = {power(3, 3)}")

# TODO: Create function with default parameter for discount


print("\n=== KEYWORD ARGUMENTS ===\n")

def create_profile(name, age, city, country):
    """Create a user profile"""
    return f"{name}, {age}, from {city}, {country}"

# Positional arguments
print(create_profile("Alice", 25, "Paris", "France"))

# Keyword arguments (order doesn't matter)
print(create_profile(city="London", name="Bob", country="UK", age=30))

# Mix of both (positional first)
print(create_profile("Charlie", 35, country="USA", city="NYC"))

# TODO: Create function and call it using keyword arguments


print("\n=== DOCSTRINGS ===\n")

def calculate_area(length, width):
    """
    Calculate the area of a rectangle.

    Parameters:
        length (float): The length of the rectangle
        width (float): The width of the rectangle

    Returns:
        float: The area of the rectangle
    """
    return length * width

# Access docstring
print(calculate_area.__doc__)

# Use help()
help(calculate_area)

# TODO: Write a function with comprehensive docstring


print("\n=== PRACTICE EXERCISES ===\n")

# Exercise 1: Simple Functions
# TODO: Create function that prints "Python is awesome!"


# Exercise 2: Greeting Function
# TODO: Create function that takes name and time of day
# Returns "Good morning, Alice!" or "Good evening, Bob!"


# Exercise 3: Even or Odd
# TODO: Create function that returns True if number is even


# Exercise 4: Maximum of Three
# TODO: Create function that returns the largest of 3 numbers


# Exercise 5: Temperature Converter
# TODO: Create function to convert Celsius to Fahrenheit


# Exercise 6: String Reverser
# TODO: Create function that returns reversed string


# Exercise 7: Count Vowels
# TODO: Create function that counts vowels in a string


# Exercise 8: Factorial
# TODO: Create function that calculates factorial


print("\n=== BONUS CHALLENGES ===\n")

# Challenge 1: Palindrome Checker
# TODO: Function that checks if string is palindrome


# Challenge 2: Prime Checker
# TODO: Function that checks if number is prime


# Challenge 3: FizzBuzz Function
# TODO: Function that returns "Fizz", "Buzz", "FizzBuzz", or the number


print("\nGreat start with functions! Next: 17_function_parameters.py")
