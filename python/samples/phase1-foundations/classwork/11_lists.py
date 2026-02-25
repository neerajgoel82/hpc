"""
Exercise 11: Lists

Learning Objectives:
- Creating and accessing lists
- List methods
- List slicing
- List operations
- Nested lists
"""

print("=== CREATING LISTS ===\n")

# Empty list
empty_list = []
also_empty = list()

# List with values
numbers = [1, 2, 3, 4, 5]
fruits = ["apple", "banana", "cherry"]
mixed = [1, "hello", 3.14, True]

print(f"Numbers: {numbers}")
print(f"Fruits: {fruits}")
print(f"Mixed: {mixed}")

# TODO: Create a list of your favorite 5 movies


print("\n=== ACCESSING ELEMENTS ===\n")

colors = ["red", "green", "blue", "yellow", "purple"]

# Indexing (starts at 0)
print(f"First: {colors[0]}")
print(f"Third: {colors[2]}")
print(f"Last: {colors[-1]}")
print(f"Second to last: {colors[-2]}")

# Changing elements
colors[1] = "orange"
print(f"Modified: {colors}")

# TODO: Create a list of 5 numbers and change the middle one


print("\n=== LIST SLICING ===\n")

numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

print(f"First 5: {numbers[:5]}")
print(f"Last 5: {numbers[-5:]}")
print(f"Middle: {numbers[3:7]}")
print(f"Every 2nd: {numbers[::2]}")
print(f"Reverse: {numbers[::-1]}")
print(f"Every 3rd in reverse: {numbers[::-3]}")

# TODO: From [10, 20, 30, 40, 50, 60, 70, 80], get:
# - First 3 elements
# - Last 2 elements
# - Elements from index 2 to 5


print("\n=== LIST METHODS ===\n")

fruits = ["apple", "banana"]

# Adding elements
fruits.append("cherry")  # Add to end
print(f"After append: {fruits}")

fruits.insert(1, "orange")  # Insert at position
print(f"After insert: {fruits}")

fruits.extend(["grape", "mango"])  # Add multiple
print(f"After extend: {fruits}")

# Removing elements
fruits.remove("banana")  # Remove first occurrence
print(f"After remove: {fruits}")

popped = fruits.pop()  # Remove and return last
print(f"Popped: {popped}, List: {fruits}")

popped = fruits.pop(0)  # Remove and return at index
print(f"Popped: {popped}, List: {fruits}")

# TODO: Create list [1,2,3], add 4 and 5, remove 2


print("\n=== MORE LIST METHODS ===\n")

numbers = [3, 1, 4, 1, 5, 9, 2, 6]

# Finding elements
print(f"Index of 4: {numbers.index(4)}")
print(f"Count of 1: {numbers.count(1)}")

# Sorting
numbers.sort()  # Modifies original list
print(f"Sorted: {numbers}")

numbers.sort(reverse=True)
print(f"Reverse sorted: {numbers}")

# Using sorted() - returns new list
numbers = [3, 1, 4, 1, 5]
sorted_numbers = sorted(numbers)
print(f"Original: {numbers}")
print(f"Sorted copy: {sorted_numbers}")

# Reversing
numbers.reverse()
print(f"Reversed: {numbers}")

# Clearing
numbers_copy = numbers.copy()
numbers_copy.clear()
print(f"Cleared: {numbers_copy}")

# TODO: Sort list of names alphabetically


print("\n=== LIST OPERATIONS ===\n")

# Concatenation
list1 = [1, 2, 3]
list2 = [4, 5, 6]
combined = list1 + list2
print(f"Combined: {combined}")

# Repetition
repeated = [0] * 5
print(f"Repeated: {repeated}")

pattern = [1, 2] * 3
print(f"Pattern: {pattern}")

# Membership testing
fruits = ["apple", "banana", "cherry"]
print(f"'apple' in fruits: {'apple' in fruits}")
print(f"'grape' in fruits: {'grape' in fruits}")

# Length
print(f"Length: {len(fruits)}")

# TODO: Combine two lists and check if 'python' is in result


print("\n=== LIST COMPREHENSIONS ===\n")

# Creating lists with comprehensions
squares = [x**2 for x in range(1, 6)]
print(f"Squares: {squares}")

evens = [x for x in range(1, 11) if x % 2 == 0]
print(f"Evens: {evens}")

words = ["hello", "world", "python"]
upper_words = [word.upper() for word in words]
print(f"Uppercase: {upper_words}")

# TODO: Create list of cubes of odd numbers from 1 to 10


print("\n=== NESTED LISTS ===\n")

# List of lists
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

print(f"Matrix: {matrix}")
print(f"First row: {matrix[0]}")
print(f"Element [1][2]: {matrix[1][2]}")  # Row 1, Col 2 = 6

# Iterating nested lists
for row in matrix:
    for item in row:
        print(item, end=" ")
    print()

# TODO: Create a 3x3 tic-tac-toe board (use '_' for empty)


print("\n=== COMMON PATTERNS ===\n")

# Finding max/min
numbers = [23, 45, 12, 67, 34, 89]
print(f"Max: {max(numbers)}")
print(f"Min: {min(numbers)}")
print(f"Sum: {sum(numbers)}")

# All/Any
values = [True, True, True]
print(f"All True: {all(values)}")

values2 = [True, False, True]
print(f"Any True: {any(values2)}")

# Filtering
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
evens = [x for x in numbers if x % 2 == 0]
print(f"Even numbers: {evens}")

# Mapping
numbers = [1, 2, 3, 4, 5]
doubled = [x * 2 for x in numbers]
print(f"Doubled: {doubled}")

# TODO: From [1,2,3,4,5,6,7,8,9,10], get numbers divisible by 3


print("\n=== PRACTICE EXERCISES ===\n")

# Exercise 1: List Manipulation
print("--- List Manipulation ---")
numbers = [10, 20, 30, 40, 50]
# TODO:
# - Add 60 to the end
# - Insert 15 at index 1
# - Remove 30
# - Print the final list


# Exercise 2: Remove Duplicates
print("\n--- Remove Duplicates ---")
numbers = [1, 2, 2, 3, 4, 4, 5, 1, 6]
# TODO: Create new list without duplicates (keep order)


# Exercise 3: Merge and Sort
print("\n--- Merge and Sort ---")
list1 = [5, 2, 8, 1]
list2 = [7, 3, 9, 4]
# TODO: Combine both lists and sort in descending order


# Exercise 4: List Statistics
print("\n--- Statistics ---")
scores = [85, 92, 78, 90, 88, 76, 95, 89]
# TODO: Calculate:
# - Average
# - Highest score
# - Lowest score
# - Scores above average


# Exercise 5: Second Largest
print("\n--- Second Largest ---")
numbers = [45, 23, 67, 89, 12, 34, 56]
# TODO: Find second largest number (without sorting)


# Exercise 6: Rotate List
print("\n--- Rotate List ---")
numbers = [1, 2, 3, 4, 5]
# TODO: Rotate right by 2 positions
# Expected: [4, 5, 1, 2, 3]


# Exercise 7: List Intersection
print("\n--- Common Elements ---")
list1 = [1, 2, 3, 4, 5]
list2 = [3, 4, 5, 6, 7]
# TODO: Find elements common to both


# Exercise 8: Flatten Nested List
print("\n--- Flatten ---")
nested = [[1, 2], [3, 4], [5, 6]]
# TODO: Create single flat list [1,2,3,4,5,6]


print("\n=== BONUS CHALLENGES ===\n")

# Challenge 1: Running Sum
numbers = [1, 2, 3, 4, 5]
# TODO: Create list of running sums
# Expected: [1, 3, 6, 10, 15]


# Challenge 2: Chunk List
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# TODO: Split into chunks of size 3
# Expected: [[1,2,3], [4,5,6], [7,8,9]]


# Challenge 3: Matrix Transpose
matrix = [
    [1, 2, 3],
    [4, 5, 6]
]
# TODO: Transpose (swap rows and columns)
# Expected: [[1, 4], [2, 5], [3, 6]]


print("\nExcellent! Next: 12_tuples.py")
