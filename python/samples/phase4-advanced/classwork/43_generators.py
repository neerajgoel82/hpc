"""
Exercise 43: Generators

Learning Objectives:
- yield keyword
- Generator functions
- Generator expressions
- Benefits of generators
- yield from
"""

print("=== GENERATOR BASICS ===\n")

# Regular function returns all at once
def regular_range(n):
    """Returns list of numbers"""
    result = []
    i = 0
    while i < n:
        result.append(i)
        i += 1
    return result

# Generator function yields one at a time
def generator_range(n):
    """Yields numbers one at a time"""
    i = 0
    while i < n:
        yield i  # Pause here, return value, resume next time
        i += 1

# Compare memory usage
regular = regular_range(5)
gen = generator_range(5)

print(f"Regular function: {regular}")  # List
print(f"Generator: {gen}")  # Generator object

# Iterate generator
print("Generator values:")
for num in gen:
    print(num)

# TODO: Create generator for even numbers


print("\n=== GENERATOR EXPRESSIONS ===\n")

# List comprehension
squares_list = [x**2 for x in range(5)]
print(f"List: {squares_list}")

# Generator expression (use parentheses)
squares_gen = (x**2 for x in range(5))
print(f"Generator: {squares_gen}")

# Convert to list
print(f"Gen to list: {list(squares_gen)}")

# One-time use
gen = (x**2 for x in range(5))
print(f"First iteration: {list(gen)}")
print(f"Second iteration: {list(gen)}")  # Empty!

# TODO: Create generator expression for cubes


print("\n=== PRACTICAL EXAMPLES ===\n")

# Fibonacci generator
def fibonacci(n):
    """Generate first n Fibonacci numbers"""
    a, b = 0, 1
    count = 0
    while count < n:
        yield a
        a, b = b, a + b
        count += 1

print("Fibonacci sequence:")
for num in fibonacci(10):
    print(num, end=" ")
print()

# Infinite generator
def infinite_counter():
    """Count forever!"""
    n = 0
    while True:
        yield n
        n += 1

counter = infinite_counter()
print(f"First 5 from infinite: {[next(counter) for _ in range(5)]}")

# TODO: Create infinite sequence of even numbers


print("\n=== READING FILES WITH GENERATORS ===\n")

def read_large_file(file_path):
    """Read file line by line (memory efficient)"""
    with open(file_path) as file:
        for line in file:
            yield line.strip()

# Example usage (would work with real file)
# for line in read_large_file("huge_file.txt"):
#     process(line)

# TODO: Create generator to read CSV rows


print("\n=== YIELD FROM ===\n")

def generator1():
    yield 1
    yield 2
    yield 3

def generator2():
    yield 'a'
    yield 'b'
    yield 'c'

def combined():
    """Combine multiple generators"""
    yield from generator1()
    yield from generator2()

print("Combined:", list(combined()))

# Flatten nested list
def flatten(nested_list):
    """Flatten arbitrarily nested list"""
    for item in nested_list:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item

nested = [1, [2, 3, [4, 5]], 6, [7, [8, 9]]]
print(f"Flattened: {list(flatten(nested))}")

# TODO: Create generator that chains multiple iterables


print("\n=== BENEFITS OF GENERATORS ===\n")

# 1. Memory efficient
def large_range(n):
    """Generator: O(1) memory"""
    for i in range(n):
        yield i

# vs list: O(n) memory

# 2. Lazy evaluation
def expensive_operation():
    """Only compute when needed"""
    for i in range(1000000):
        # Expensive calculation
        result = i ** 2
        yield result

# Only computes first 5
gen = expensive_operation()
first_five = [next(gen) for _ in range(5)]
print(f"First 5 squares: {first_five}")

# 3. Infinite sequences
def powers_of_two():
    """Infinite sequence of powers of 2"""
    n = 0
    while True:
        yield 2 ** n
        n += 1

# TODO: Understand when to use generators vs lists


print("\n=== PRACTICE EXERCISES ===\n")

# Exercise 1: Range Generator
# TODO: Implement your own range() generator


# Exercise 2: Prime Generator
# TODO: Generator that yields prime numbers


# Exercise 3: File Line Counter
# TODO: Generator to count lines in large file


# Exercise 4: Batch Generator
# TODO: Generator that yields items in batches
# Example: [1,2,3,4,5,6] with batch_size=2 -> [1,2], [3,4], [5,6]


# Exercise 5: Running Average
# TODO: Generator that yields running average


print("\n=== BONUS CHALLENGES ===\n")

# Challenge 1: Generator Pipeline
# TODO: Chain multiple generators to process data
# Example: read_file -> parse_line -> filter -> transform


# Challenge 2: Cyclic Generator
# TODO: Generator that cycles through items forever


# Challenge 3: Look-ahead Generator
# TODO: Generator that can peek at next value


print("\nAwesome! Next: 44_decorators_basics.py")
