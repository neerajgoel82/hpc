"""
Exercise 14: Sets

Learning Objectives:
- Creating sets
- Set operations (union, intersection, difference)
- Set methods
- When to use sets
- Frozen sets
"""

print("=== CREATING SETS ===\n")

# Empty set (must use set(), not {})
empty = set()
not_empty = {1, 2, 3}  # This creates a set
empty_dict = {}  # This creates a dict!

# Set with values
numbers = {1, 2, 3, 4, 5}
fruits = {"apple", "banana", "cherry"}

print(f"Numbers: {numbers}")
print(f"Fruits: {fruits}")

# Sets automatically remove duplicates
duplicates = {1, 2, 2, 3, 3, 3, 4}
print(f"Set removes duplicates: {duplicates}")

# Create set from list
list_with_dupes = [1, 2, 2, 3, 3, 4]
unique = set(list_with_dupes)
print(f"Unique values: {unique}")

# TODO: Create set from string to see unique characters


print("\n=== SET CHARACTERISTICS ===\n")

# Unordered (no indexing)
my_set = {3, 1, 4, 1, 5, 9}
print(f"Set: {my_set}")
# print(my_set[0])  # TypeError: 'set' object is not subscriptable

# Mutable (can add/remove items)
my_set.add(2)
print(f"After add: {my_set}")

# Items must be immutable (hashable)
# valid_set = {1, "hello", (1, 2)}  # OK
# invalid_set = {1, [2, 3]}  # TypeError: unhashable type: 'list'

# TODO: Understand why sets are useful for removing duplicates


print("\n=== ADDING & REMOVING ===\n")

colors = {"red", "green", "blue"}

# Add single element
colors.add("yellow")
print(f"After add: {colors}")

colors.add("red")  # Adding duplicate has no effect
print(f"Add duplicate: {colors}")

# Add multiple elements
colors.update(["purple", "orange", "pink"])
print(f"After update: {colors}")

# Remove element (raises error if not found)
colors.remove("pink")
print(f"After remove: {colors}")

# Discard (no error if not found)
colors.discard("brown")  # No error even though brown isn't in set
print(f"After discard: {colors}")

# Pop (removes arbitrary element)
popped = colors.pop()
print(f"Popped: {popped}, Remaining: {colors}")

# Clear all
# colors.clear()

# TODO: Practice adding and removing elements


print("\n=== SET OPERATIONS ===\n")

set1 = {1, 2, 3, 4, 5}
set2 = {4, 5, 6, 7, 8}

# Union (all elements from both sets)
union = set1 | set2
print(f"Union {set1} | {set2} = {union}")
print(f"Union method: {set1.union(set2)}")

# Intersection (elements in both sets)
intersection = set1 & set2
print(f"Intersection {set1} & {set2} = {intersection}")
print(f"Intersection method: {set1.intersection(set2)}")

# Difference (elements in first but not second)
difference = set1 - set2
print(f"Difference {set1} - {set2} = {difference}")
print(f"Difference method: {set1.difference(set2)}")

# Symmetric difference (elements in either but not both)
sym_diff = set1 ^ set2
print(f"Symmetric diff {set1} ^ {set2} = {sym_diff}")
print(f"Symmetric diff method: {set1.symmetric_difference(set2)}")

# TODO: Find common elements between {1,2,3,4} and {3,4,5,6}


print("\n=== SET COMPARISONS ===\n")

set_a = {1, 2, 3}
set_b = {1, 2, 3, 4, 5}
set_c = {1, 2, 3}

# Subset (all elements of A are in B)
print(f"{set_a} is subset of {set_b}: {set_a <= set_b}")
print(f"{set_a} is subset of {set_b}: {set_a.issubset(set_b)}")

# Proper subset (subset but not equal)
print(f"{set_a} is proper subset of {set_b}: {set_a < set_b}")

# Superset (A contains all elements of B)
print(f"{set_b} is superset of {set_a}: {set_b >= set_a}")
print(f"{set_b} is superset of {set_a}: {set_b.issuperset(set_a)}")

# Equal sets
print(f"{set_a} == {set_c}: {set_a == set_c}")

# Disjoint (no common elements)
set_x = {1, 2, 3}
set_y = {4, 5, 6}
print(f"{set_x} and {set_y} are disjoint: {set_x.isdisjoint(set_y)}")

# TODO: Check if {1,2} is subset of {1,2,3,4,5}


print("\n=== ITERATING SETS ===\n")

fruits = {"apple", "banana", "cherry", "date"}

# Basic iteration (order not guaranteed)
for fruit in fruits:
    print(fruit)

# Sorted iteration
for fruit in sorted(fruits):
    print(fruit)

# With enumerate
for i, fruit in enumerate(sorted(fruits), 1):
    print(f"{i}. {fruit}")

# TODO: Iterate and print set elements in reverse alphabetical order


print("\n=== SET COMPREHENSION ===\n")

# Create set with comprehension
squares = {x**2 for x in range(1, 6)}
print(f"Squares: {squares}")

# With condition
even_squares = {x**2 for x in range(1, 11) if x % 2 == 0}
print(f"Even squares: {even_squares}")

# From string
text = "hello world"
unique_chars = {char for char in text if char.isalpha()}
print(f"Unique characters: {unique_chars}")

# TODO: Create set of even numbers from 1 to 20


print("\n=== FROZEN SETS ===\n")

# Immutable set (can be used as dict key or set element)
frozen = frozenset([1, 2, 3, 4, 5])
print(f"Frozen set: {frozen}")

# Can't modify
# frozen.add(6)  # AttributeError

# Can use as dict key
lookup = {frozen: "group1"}
print(f"Using frozen set as key: {lookup}")

# TODO: Create frozenset and try to use it as dict key


print("\n=== COMMON USE CASES ===\n")

# 1. Remove duplicates from list
numbers = [1, 2, 2, 3, 3, 3, 4, 4, 5]
unique = list(set(numbers))
print(f"Unique: {unique}")

# 2. Membership testing (very fast)
valid_usernames = {"alice", "bob", "charlie"}
username = "alice"
if username in valid_usernames:  # O(1) average case
    print(f"{username} is valid")

# 3. Finding unique elements
list1 = [1, 2, 3, 4]
list2 = [3, 4, 5, 6]
unique_to_list1 = set(list1) - set(list2)
print(f"Unique to list1: {unique_to_list1}")

# 4. Find common elements
common = set(list1) & set(list2)
print(f"Common elements: {common}")

# TODO: Use set to find duplicates in a list


print("\n=== PRACTICE EXERCISES ===\n")

# Exercise 1: Remove Duplicates
print("--- Remove Duplicates ---")
data = [1, 2, 2, 3, 4, 4, 5, 5, 5]
# TODO: Remove duplicates and sort


# Exercise 2: Common Students
print("\n--- Common Students ---")
class_a = {"Alice", "Bob", "Charlie", "Diana"}
class_b = {"Bob", "Diana", "Eve", "Frank"}
# TODO: Find students in both classes
# TODO: Find students only in class A
# TODO: Find all unique students


# Exercise 3: Valid Characters
print("\n--- Character Validation ---")
allowed = set("abcdefghijklmnopqrstuvwxyz0123456789")
password = "abc123xyz"
# TODO: Check if password contains only allowed characters


# Exercise 4: Set Operations
print("\n--- Set Operations ---")
set1 = {1, 2, 3, 4, 5}
set2 = {4, 5, 6, 7, 8}
set3 = {5, 6, 7, 8, 9}
# TODO: Find elements common to all three sets
# TODO: Find elements unique to set1


# Exercise 5: Unique Words
print("\n--- Unique Words ---")
sentence = "the quick brown fox jumps over the lazy dog"
# TODO: Find all unique words
# TODO: Count how many unique words


# Exercise 6: Vowels and Consonants
print("\n--- Vowels and Consonants ---")
text = "Hello World"
# TODO: Find all unique vowels
# TODO: Find all unique consonants


# Exercise 7: Missing Numbers
print("\n--- Missing Numbers ---")
complete = set(range(1, 11))
present = {1, 3, 5, 7, 9}
# TODO: Find missing numbers


# Exercise 8: Symmetric Difference Application
print("\n--- Changed Files ---")
old_files = {"file1.txt", "file2.txt", "file3.txt"}
new_files = {"file2.txt", "file3.txt", "file4.txt"}
# TODO: Find files that were added or removed


print("\n=== BONUS CHALLENGES ===\n")

# Challenge 1: Pangram Checker
# A pangram contains every letter of the alphabet
sentence = "the quick brown fox jumps over the lazy dog"
# TODO: Check if sentence is a pangram


# Challenge 2: First Duplicate
numbers = [1, 2, 3, 4, 2, 5, 6, 3]
# TODO: Find the first number that appears twice


# Challenge 3: Power Set
# Power set = all possible subsets
original = {1, 2, 3}
# TODO: Generate all subsets
# Expected: {}, {1}, {2}, {3}, {1,2}, {1,3}, {2,3}, {1,2,3}


print("\nGreat! Next: 15_list_comprehensions.py")
