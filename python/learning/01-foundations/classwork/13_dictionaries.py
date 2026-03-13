"""
Exercise 13: Dictionaries

Learning Objectives:
- Creating dictionaries
- Accessing and modifying values
- Dictionary methods
- Iterating over dictionaries
- Nested dictionaries
"""

print("=== CREATING DICTIONARIES ===\n")

# Empty dictionary
empty = {}
also_empty = dict()

# Dictionary with key-value pairs
student = {
    "name": "Alice",
    "age": 20,
    "grade": "A"
}

# Different data types
mixed = {
    "string": "hello",
    "number": 42,
    "list": [1, 2, 3],
    "boolean": True
}

print(f"Student: {student}")
print(f"Mixed: {mixed}")

# TODO: Create dictionary for a book (title, author, year, pages)


print("\n=== ACCESSING VALUES ===\n")

person = {
    "name": "Bob",
    "age": 30,
    "city": "London"
}

# Using square brackets
print(f"Name: {person['name']}")
print(f"Age: {person['age']}")

# Using get() method (safer)
print(f"City: {person.get('city')}")
print(f"Country: {person.get('country')}")  # Returns None if not found
print(f"Country: {person.get('country', 'Unknown')}")  # Default value

# TODO: Create dict and safely get a non-existent key


print("\n=== ADDING & MODIFYING ===\n")

car = {"brand": "Toyota", "year": 2020}
print(f"Original: {car}")

# Add new key-value pair
car["color"] = "blue"
print(f"After adding color: {car}")

# Modify existing value
car["year"] = 2021
print(f"After modifying year: {car}")

# Update multiple values
car.update({"year": 2022, "price": 25000})
print(f"After update: {car}")

# TODO: Create dict, add 3 new keys, modify 1 existing


print("\n=== REMOVING ENTRIES ===\n")

inventory = {"apples": 5, "bananas": 3, "oranges": 7}

# Remove specific key
del inventory["bananas"]
print(f"After del: {inventory}")

# Pop: remove and return value
apples = inventory.pop("apples")
print(f"Popped: {apples}, Remaining: {inventory}")

# Pop with default (no error if key doesn't exist)
value = inventory.pop("grapes", 0)
print(f"Popped grapes: {value}")

# Remove last inserted item
inventory["bananas"] = 3
inventory["grapes"] = 4
last = inventory.popitem()
print(f"Last item: {last}, Remaining: {inventory}")

# Clear all
# inventory.clear()

# TODO: Practice removing keys safely


print("\n=== DICTIONARY METHODS ===\n")

data = {"a": 1, "b": 2, "c": 3}

# Keys, values, items
print(f"Keys: {data.keys()}")
print(f"Values: {data.values()}")
print(f"Items: {data.items()}")

# Convert to lists
keys_list = list(data.keys())
values_list = list(data.values())
print(f"Keys as list: {keys_list}")

# Check if key exists
print(f"'a' in dict: {'a' in data}")
print(f"'z' in dict: {'z' in data}")

# Get all keys
print(f"All keys: {', '.join(data.keys())}")

# TODO: Create dict and check if specific keys exist


print("\n=== ITERATING DICTIONARIES ===\n")

scores = {"Alice": 85, "Bob": 92, "Charlie": 78}

# Iterate over keys (default)
print("Keys:")
for key in scores:
    print(key)

# Iterate over values
print("\nValues:")
for value in scores.values():
    print(value)

# Iterate over key-value pairs
print("\nKey-Value pairs:")
for name, score in scores.items():
    print(f"{name}: {score}")

# With enumerate
print("\nEnumerated:")
for i, (name, score) in enumerate(scores.items(), 1):
    print(f"{i}. {name}: {score}")

# TODO: Iterate over a dict and print formatted output


print("\n=== DICTIONARY COMPREHENSION ===\n")

# Create dictionary with comprehension
squares = {x: x**2 for x in range(1, 6)}
print(f"Squares: {squares}")

# With condition
even_squares = {x: x**2 for x in range(1, 11) if x % 2 == 0}
print(f"Even squares: {even_squares}")

# Transform existing dict
prices = {"apple": 2.5, "banana": 1.5, "cherry": 3.0}
discounted = {item: price * 0.9 for item, price in prices.items()}
print(f"Discounted: {discounted}")

# TODO: Create dict of numbers 1-10 with their cubes


print("\n=== NESTED DICTIONARIES ===\n")

# Dictionary containing dictionaries
students = {
    "student1": {
        "name": "Alice",
        "age": 20,
        "grades": [85, 90, 88]
    },
    "student2": {
        "name": "Bob",
        "age": 21,
        "grades": [78, 85, 80]
    }
}

print(f"Student1 name: {students['student1']['name']}")
print(f"Student2 grades: {students['student2']['grades']}")

# Iterate nested dict
for student_id, info in students.items():
    print(f"\n{student_id}:")
    for key, value in info.items():
        print(f"  {key}: {value}")

# TODO: Create nested dict for a library (books with details)


print("\n=== COMMON PATTERNS ===\n")

# Counting occurrences
text = "hello world"
char_count = {}
for char in text:
    if char in char_count:
        char_count[char] += 1
    else:
        char_count[char] = 1
print(f"Character count: {char_count}")

# Better way with get()
text = "hello world"
char_count = {}
for char in text:
    char_count[char] = char_count.get(char, 0) + 1
print(f"Character count: {char_count}")

# Grouping data
students = [
    {"name": "Alice", "grade": "A"},
    {"name": "Bob", "grade": "B"},
    {"name": "Charlie", "grade": "A"},
    {"name": "Diana", "grade": "B"}
]

by_grade = {}
for student in students:
    grade = student["grade"]
    if grade not in by_grade:
        by_grade[grade] = []
    by_grade[grade].append(student["name"])
print(f"Grouped by grade: {by_grade}")

# TODO: Count word frequency in a sentence


print("\n=== DEFAULT DICT ===\n")

from collections import defaultdict

# Regular dict would raise KeyError
word_count = defaultdict(int)  # Default value is 0
for word in ["apple", "banana", "apple", "cherry", "banana", "apple"]:
    word_count[word] += 1  # No need to check if key exists
print(f"Word count: {dict(word_count)}")

# With list as default
groups = defaultdict(list)
groups["fruits"].append("apple")
groups["fruits"].append("banana")
groups["vegetables"].append("carrot")
print(f"Groups: {dict(groups)}")

# TODO: Use defaultdict to group numbers by whether they're even or odd


print("\n=== PRACTICE EXERCISES ===\n")

# Exercise 1: Phone Book
print("--- Phone Book ---")
# TODO: Create phone book dict
# Add entries, lookup by name, update numbers


# Exercise 2: Inventory System
print("\n--- Inventory ---")
inventory = {"apple": 50, "banana": 30, "cherry": 20}
# TODO:
# - Add new items
# - Update quantities
# - Remove out-of-stock items (quantity 0)
# - Calculate total items


# Exercise 3: Grade Calculator
print("\n--- Grade Calculator ---")
students = {
    "Alice": [85, 90, 88],
    "Bob": [78, 85, 80],
    "Charlie": [92, 88, 95]
}
# TODO: Calculate average grade for each student


# Exercise 4: Word Frequency
print("\n--- Word Frequency ---")
sentence = "the quick brown fox jumps over the lazy dog the fox"
# TODO: Count how many times each word appears


# Exercise 5: Merge Dictionaries
print("\n--- Merge Dicts ---")
dict1 = {"a": 1, "b": 2}
dict2 = {"b": 3, "c": 4}
# TODO: Merge dicts (dict2 values override dict1)


# Exercise 6: Reverse Dictionary
print("\n--- Reverse Dict ---")
original = {"a": 1, "b": 2, "c": 3}
# TODO: Create new dict with values as keys
# Expected: {1: "a", 2: "b", 3: "c"}


# Exercise 7: Filter Dictionary
print("\n--- Filter Dict ---")
scores = {"Alice": 85, "Bob": 92, "Charlie": 78, "Diana": 95}
# TODO: Create new dict with only scores >= 90


# Exercise 8: Nested Access
print("\n--- Nested Dict ---")
data = {
    "company": {
        "name": "TechCorp",
        "employees": {
            "engineering": 50,
            "sales": 30
        }
    }
}
# TODO: Safely access nested values (handle missing keys)


print("\n=== BONUS CHALLENGES ===\n")

# Challenge 1: Dictionary from Lists
names = ["Alice", "Bob", "Charlie"]
scores = [85, 92, 78]
# TODO: Create dict combining both lists


# Challenge 2: Most Common Element
numbers = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
# TODO: Find the most common number


# Challenge 3: Group Anagrams
words = ["eat", "tea", "tan", "ate", "nat", "bat"]
# TODO: Group words that are anagrams of each other
# Expected: {("eat", "tea", "ate"), ("tan", "nat"), ("bat",)}


print("\nAwesome! Next: 14_sets.py")
