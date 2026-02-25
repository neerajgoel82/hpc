"""
Exercise 2: Variables and Data Types

Learning Objectives:
- Create and use variables
- Understand different data types (int, float, str, bool)
- Type conversion
- Variable naming conventions
"""

# Variables store data that can be used later
# Variable naming: use lowercase with underscores (snake_case)

# Integers (whole numbers)
age = 25
year = 2024
temperature = -10

print("Age:", age)
print("Year:", year)

# TODO: Create a variable for your age and print it


# Floats (decimal numbers)
price = 19.99
pi = 3.14159
temperature_celsius = 36.6

print("Price:", price)

# TODO: Create a variable for your height in meters and print it


# Strings (text)
name = "Alice"
greeting = "Hello"
message = 'Single quotes work too!'

print(greeting, name)

# TODO: Create variables for your first and last name, then print them


# Booleans (True or False)
is_student = True
is_raining = False
has_pet = True

print("Is student:", is_student)

# TODO: Create a boolean variable for whether you like Python and print it


# Type checking with type()
print("\n--- Checking Types ---")
print(type(age))          # <class 'int'>
print(type(price))        # <class 'float'>
print(type(name))         # <class 'str'>
print(type(is_student))   # <class 'bool'>

# TODO: Check the type of your variables


# Type conversion
print("\n--- Type Conversion ---")
num_string = "42"
num_int = int(num_string)        # String to integer
print(num_int + 8)               # 50

float_num = float(num_string)    # String to float
print(float_num)                 # 42.0

age_string = str(age)            # Integer to string
print("I am " + age_string + " years old")

# TODO: Create a string "123" and convert it to an integer, then multiply by 2


# Multiple assignment
x, y, z = 1, 2, 3
print(x, y, z)

# Same value to multiple variables
a = b = c = 0
print(a, b, c)

# TODO: Create three variables (color1, color2, color3) in one line with different color names


# Constants (convention: use UPPERCASE)
PI = 3.14159
MAX_USERS = 100
COMPANY_NAME = "TechCorp"

print("\n--- Constants ---")
print("PI:", PI)

# TODO: Create a constant for the number of days in a week


# Quick Challenge: Variable swap
print("\n--- Challenge: Swap Variables ---")
x = 5
y = 10
print("Before swap: x =", x, "y =", y)

# TODO: Swap the values of x and y (x should be 10, y should be 5)
# Hint: You'll need a temporary variable

# print("After swap: x =", x, "y =", y)


# Practice Problems
print("\n--- Practice Problems ---")

# 1. Store your birth year in a variable and calculate your age
birth_year = 2000  # TODO: Change this to your birth year
current_year = 2024
# TODO: Calculate and print your age


# 2. Calculate the area of a rectangle
length = 10
width = 5
# TODO: Calculate area and print it


# 3. Create a personalized greeting
user_name = "Bob"  # TODO: Change to your name
# TODO: Print "Welcome, [name]! Let's learn Python together."


print("\nGreat job! Move on to 03_operators.py when ready.")
