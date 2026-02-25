"""
Exercise 12: Tuples

Learning Objectives:
- Creating tuples
- Tuple immutability
- Tuple operations
- When to use tuples vs lists
- Tuple unpacking
"""

print("=== CREATING TUPLES ===\n")

# Empty tuple
empty = ()
also_empty = tuple()

# Tuple with values
numbers = (1, 2, 3, 4, 5)
fruits = ("apple", "banana", "cherry")
mixed = (1, "hello", 3.14, True)

print(f"Numbers: {numbers}")
print(f"Fruits: {fruits}")

# Single element tuple (note the comma!)
single = (42,)  # With comma: tuple
not_tuple = (42)  # Without comma: just an integer
print(f"Single element: {single}, type: {type(single)}")
print(f"Without comma: {not_tuple}, type: {type(not_tuple)}")

# Without parentheses (tuple packing)
coordinates = 10, 20
print(f"Coordinates: {coordinates}, type: {type(coordinates)}")

# TODO: Create a tuple with your name, age, and city


print("\n=== ACCESSING TUPLE ELEMENTS ===\n")

colors = ("red", "green", "blue", "yellow", "purple")

# Indexing (same as lists)
print(f"First: {colors[0]}")
print(f"Last: {colors[-1]}")
print(f"Third: {colors[2]}")

# Slicing (same as lists)
print(f"First 3: {colors[:3]}")
print(f"Last 2: {colors[-2:]}")
print(f"Reversed: {colors[::-1]}")

# TODO: From ('a', 'b', 'c', 'd', 'e'), get middle 3 elements


print("\n=== TUPLE IMMUTABILITY ===\n")

# Tuples cannot be modified after creation
point = (10, 20)
print(f"Point: {point}")

# This will cause an error:
# point[0] = 15  # TypeError: 'tuple' object does not support item assignment

# But you can create a new tuple
point = (15, 25)
print(f"New point: {point}")

# Tuples with mutable objects
mixed = (1, 2, [3, 4])
mixed[2].append(5)  # The list inside can be modified
print(f"Modified: {mixed}")

# TODO: Try to understand why immutability is useful


print("\n=== TUPLE OPERATIONS ===\n")

tuple1 = (1, 2, 3)
tuple2 = (4, 5, 6)

# Concatenation
combined = tuple1 + tuple2
print(f"Combined: {combined}")

# Repetition
repeated = (0,) * 5
print(f"Repeated: {repeated}")

# Membership
print(f"2 in tuple1: {2 in tuple1}")
print(f"10 in tuple1: {10 in tuple1}")

# Length
print(f"Length: {len(combined)}")

# TODO: Create two tuples and combine them


print("\n=== TUPLE METHODS ===\n")

# Tuples have only 2 methods (because they're immutable)
numbers = (1, 2, 3, 2, 4, 2, 5)

# count() - count occurrences
print(f"Count of 2: {numbers.count(2)}")

# index() - find first occurrence
print(f"Index of 3: {numbers.index(3)}")
# print(numbers.index(10))  # ValueError if not found

# Other useful functions
print(f"Max: {max(numbers)}")
print(f"Min: {min(numbers)}")
print(f"Sum: {sum(numbers)}")

# TODO: Count how many times 5 appears in (5, 10, 5, 20, 5, 30)


print("\n=== TUPLE UNPACKING ===\n")

# Unpack into variables
point = (10, 20)
x, y = point
print(f"x = {x}, y = {y}")

# Multiple values
person = ("Alice", 25, "Paris")
name, age, city = person
print(f"Name: {name}, Age: {age}, City: {city}")

# Using * to capture multiple values
numbers = (1, 2, 3, 4, 5)
first, *middle, last = numbers
print(f"First: {first}")
print(f"Middle: {middle}")  # This will be a list!
print(f"Last: {last}")

# Swapping variables
a, b = 10, 20
a, b = b, a  # Swap using tuple unpacking
print(f"After swap: a = {a}, b = {b}")

# TODO: Unpack (100, 200, 300) into three variables


print("\n=== TUPLE VS LIST ===\n")

# Use tuples when:
# 1. Data shouldn't change (immutable)
# 2. Returning multiple values from a function
# 3. Dictionary keys (tuples can be keys, lists cannot)
# 4. Slightly faster than lists

# Use lists when:
# 1. Data needs to be modified
# 2. Working with collections that grow/shrink

# Example: Returning multiple values
def get_coordinates():
    return (10, 20)

x, y = get_coordinates()
print(f"Coordinates: ({x}, {y})")

# Tuples as dictionary keys
locations = {
    (0, 0): "Origin",
    (1, 0): "East",
    (0, 1): "North"
}
print(f"Location at (0,0): {locations[(0, 0)]}")

# TODO: Create function that returns (min, max, average) of a list


print("\n=== NAMED TUPLES ===\n")

# For better readability, you can use named tuples
from collections import namedtuple

# Define a named tuple type
Point = namedtuple('Point', ['x', 'y'])
Color = namedtuple('Color', ['red', 'green', 'blue'])

# Create instances
p = Point(10, 20)
c = Color(255, 0, 0)

# Access by name or index
print(f"Point: x={p.x}, y={p.y}")
print(f"Point by index: {p[0]}, {p[1]}")
print(f"Color: RGB({c.red}, {c.green}, {c.blue})")

# TODO: Create a Person named tuple with name, age, city


print("\n=== PRACTICE EXERCISES ===\n")

# Exercise 1: Tuple Basics
print("--- Tuple Basics ---")
# TODO: Create tuple with days of the week
# Print first day, last day, and weekdays (Mon-Fri)


# Exercise 2: Min, Max, Average
print("\n--- Statistics ---")
numbers = (45, 23, 67, 89, 12, 34, 56)
# TODO: Find min, max, and average (use tuple unpacking to return)


# Exercise 3: Coordinate Distance
print("\n--- Distance Between Points ---")
point1 = (0, 0)
point2 = (3, 4)
# TODO: Calculate distance between two points
# Formula: sqrt((x2-x1)² + (y2-y1)²)
# Hint: Use ** for power, ** 0.5 for square root


# Exercise 4: Tuple Sorting
print("\n--- Sort Tuples ---")
students = [
    ("Alice", 85),
    ("Bob", 92),
    ("Charlie", 78),
    ("Diana", 95)
]
# TODO: Sort by score (descending)


# Exercise 5: Nested Tuples
print("\n--- Matrix Operations ---")
matrix = (
    (1, 2, 3),
    (4, 5, 6),
    (7, 8, 9)
)
# TODO: Calculate sum of all elements


# Exercise 6: Tuple Conversion
print("\n--- Convert List to Tuple ---")
my_list = [1, 2, 3, 4, 5]
# TODO: Convert to tuple, add new elements, convert back to list


# Exercise 7: Multiple Return Values
print("\n--- Function with Multiple Returns ---")
# TODO: Write function that takes a string and returns:
# (length, first_char, last_char, reversed_string)
# as a tuple


# Exercise 8: Enumerate with Tuples
print("\n--- Enumerate ---")
fruits = ("apple", "banana", "cherry", "date")
# TODO: Print each fruit with its index using enumerate


print("\n=== BONUS CHALLENGES ===\n")

# Challenge 1: Tuple Arithmetic
# TODO: Create function that takes two tuples of same length
# and returns element-wise sum
# Example: (1, 2, 3) + (4, 5, 6) = (5, 7, 9)


# Challenge 2: Find Duplicates
numbers = (1, 2, 3, 2, 4, 5, 3, 6)
# TODO: Find all duplicate values


# Challenge 3: Rotating Tuples
colors = ("red", "green", "blue", "yellow")
# TODO: Rotate right by 1
# Expected: ("yellow", "red", "green", "blue")


print("\nGreat work! Next: 13_dictionaries.py")
