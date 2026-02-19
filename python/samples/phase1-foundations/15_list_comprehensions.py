"""
Exercise 15: List Comprehensions (Advanced)

Learning Objectives:
- List comprehensions in depth
- Dictionary comprehensions
- Set comprehensions
- Nested comprehensions
- When to use comprehensions vs loops
"""

print("=== BASIC LIST COMPREHENSION ===\n")

# Traditional way
squares = []
for x in range(1, 6):
    squares.append(x ** 2)
print(f"Traditional: {squares}")

# List comprehension
squares = [x ** 2 for x in range(1, 6)]
print(f"Comprehension: {squares}")

# Syntax: [expression for item in iterable]

# More examples
evens = [x for x in range(1, 11) if x % 2 == 0]
doubled = [x * 2 for x in [1, 2, 3, 4, 5]]
upper = [word.upper() for word in ["hello", "world"]]

print(f"Evens: {evens}")
print(f"Doubled: {doubled}")
print(f"Upper: {upper}")

# TODO: Create list of cubes of numbers 1-10


print("\n=== WITH CONDITIONS ===\n")

# Filter with if
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
evens = [x for x in numbers if x % 2 == 0]
print(f"Even numbers: {evens}")

# Multiple conditions
divisible = [x for x in range(1, 21) if x % 2 == 0 if x % 3 == 0]
print(f"Divisible by 2 AND 3: {divisible}")

# If-else (conditional expression)
labels = ["even" if x % 2 == 0 else "odd" for x in range(1, 6)]
print(f"Labels: {labels}")

# Transform based on condition
absolute = [x if x >= 0 else -x for x in [-5, -3, 0, 2, 4]]
print(f"Absolute values: {absolute}")

# TODO: Get positive numbers from [-5, 3, -2, 8, -1, 6]


print("\n=== NESTED COMPREHENSIONS ===\n")

# Flatten a list
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flat = [num for row in matrix for num in row]
print(f"Flattened: {flat}")

# Create matrix
matrix = [[i * j for j in range(1, 4)] for i in range(1, 4)]
print(f"Matrix: {matrix}")

# Pairs
pairs = [(x, y) for x in [1, 2, 3] for y in [3, 4, 5]]
print(f"All pairs: {pairs}")

# With condition
pairs = [(x, y) for x in [1, 2, 3] for y in [3, 4, 5] if x != y]
print(f"Pairs where x != y: {pairs}")

# TODO: Create multiplication table (1-5) using nested comprehension


print("\n=== STRING COMPREHENSIONS ===\n")

# Characters
text = "Hello"
chars = [char for char in text]
print(f"Characters: {chars}")

# Vowels
vowels = [char for char in "Hello World" if char.lower() in "aeiou"]
print(f"Vowels: {vowels}")

# Words
sentence = "the quick brown fox"
words = [word.upper() for word in sentence.split()]
print(f"Upper words: {words}")

# First letters
first_letters = [word[0] for word in sentence.split()]
print(f"First letters: {first_letters}")

# TODO: Extract digits from "abc123def456"


print("\n=== DICTIONARY COMPREHENSION ===\n")

# Basic dictionary comprehension
squares_dict = {x: x**2 for x in range(1, 6)}
print(f"Squares dict: {squares_dict}")

# From two lists
keys = ["a", "b", "c"]
values = [1, 2, 3]
combined = {k: v for k, v in zip(keys, values)}
print(f"Combined: {combined}")

# With condition
even_squares = {x: x**2 for x in range(1, 11) if x % 2 == 0}
print(f"Even squares: {even_squares}")

# Transform existing dict
prices = {"apple": 2.5, "banana": 1.5, "cherry": 3.0}
discounted = {item: price * 0.9 for item, price in prices.items()}
print(f"Discounted: {discounted}")

# Flip dict
original = {"a": 1, "b": 2, "c": 3}
flipped = {v: k for k, v in original.items()}
print(f"Flipped: {flipped}")

# TODO: Create dict mapping numbers 1-5 to their cubes


print("\n=== SET COMPREHENSION ===\n")

# Basic set comprehension
squares_set = {x**2 for x in range(-3, 4)}
print(f"Squares set (no duplicates): {squares_set}")

# Unique characters
text = "hello world"
unique_chars = {char for char in text if char != " "}
print(f"Unique chars: {unique_chars}")

# With condition
vowels_set = {char.lower() for char in "Hello World" if char.lower() in "aeiou"}
print(f"Vowels: {vowels_set}")

# TODO: Create set of last digits of numbers 1-20


print("\n=== ADVANCED PATTERNS ===\n")

# Tuple comprehension (actually creates generator)
gen = (x**2 for x in range(5))
print(f"Generator: {gen}")
print(f"List from generator: {list(gen)}")

# Multiple transformations
numbers = [1, 2, 3, 4, 5]
result = [x ** 2 * 2 + 1 for x in numbers]
print(f"Complex transformation: {result}")

# Calling functions in comprehension
def process(x):
    return x ** 2

processed = [process(x) for x in range(1, 6)]
print(f"Function calls: {processed}")

# Filter and transform
numbers = range(1, 21)
result = [x**2 for x in numbers if x % 3 == 0]
print(f"Squares of multiples of 3: {result}")

# TODO: Double all positive numbers from [-5, 3, -2, 8, -1, 6]


print("\n=== WHEN TO USE COMPREHENSIONS ===\n")

# Good: Simple, readable transformations
squares = [x**2 for x in range(10)]

# Good: Simple filtering
evens = [x for x in range(10) if x % 2 == 0]

# Avoid: Too complex (use regular loop)
# Bad example:
# result = [x**2 if x % 2 == 0 else x**3 if x % 3 == 0 else x for x in range(10) if x > 5]
# Better as a loop with clear logic

# Avoid: Side effects
# Bad: [print(x) for x in range(10)]  # Use regular loop
# Good: for x in range(10): print(x)

# TODO: Rewrite a complex comprehension as a regular loop


print("\n=== PRACTICE EXERCISES ===\n")

# Exercise 1: Transform List
print("--- Transform ---")
temperatures_f = [32, 68, 86, 104, 122]
# TODO: Convert to Celsius using comprehension
# Formula: (F - 32) * 5/9


# Exercise 2: Filter and Transform
print("\n--- Filter and Transform ---")
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# TODO: Get squares of odd numbers


# Exercise 3: String Processing
print("\n--- String Processing ---")
words = ["hello", "world", "python", "programming"]
# TODO: Get lengths of words with comprehension


# Exercise 4: Matrix Operations
print("\n--- Matrix ---")
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# TODO: Get diagonal elements using comprehension


# Exercise 5: Dictionary from List
print("\n--- Dict from List ---")
words = ["apple", "banana", "cherry"]
# TODO: Create dict with word as key, length as value


# Exercise 6: Nested Lists
print("\n--- Nested Lists ---")
# TODO: Create list [[1], [1,2], [1,2,3], [1,2,3,4], [1,2,3,4,5]]


# Exercise 7: Filter Dictionary
print("\n--- Filter Dict ---")
scores = {"Alice": 85, "Bob": 92, "Charlie": 78, "Diana": 95, "Eve": 88}
# TODO: Get names of students who scored >= 90


# Exercise 8: Cartesian Product
print("\n--- Cartesian Product ---")
colors = ["red", "blue"]
sizes = ["S", "M", "L"]
# TODO: Create all combinations like ("red", "S"), ("red", "M"), etc.


print("\n=== BONUS CHALLENGES ===\n")

# Challenge 1: Prime Numbers
# TODO: Generate list of prime numbers from 2 to 50 using comprehension
# (Hint: a number is prime if it has no divisors from 2 to sqrt(n))


# Challenge 2: Pascal's Triangle
# TODO: Generate first 5 rows of Pascal's triangle
# Row 0: [1]
# Row 1: [1, 1]
# Row 2: [1, 2, 1]
# Row 3: [1, 3, 3, 1]
# Row 4: [1, 4, 6, 4, 1]


# Challenge 3: Flatten Nested Structures
nested = [[1, 2], [3, [4, 5]], [6, [7, [8, 9]]]]
# TODO: Flatten completely (this is advanced!)


# Challenge 4: Word Frequency
sentence = "the quick brown fox jumps over the lazy dog the fox"
# TODO: Create frequency dict using comprehension


print("\nExcellent work on Phase 1!")
print("Complete the final project: project_contact_book.py")
