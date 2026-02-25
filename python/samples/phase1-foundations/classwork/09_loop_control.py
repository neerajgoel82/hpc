"""
Exercise 9: Loop Control & Advanced Patterns

Learning Objectives:
- Master break and continue
- Pass statement
- Loop patterns and idioms
- Combining loops with conditionals
"""

print("=== BREAK STATEMENT REVIEW ===\n")

# Break exits the loop immediately
for i in range(1, 11):
    if i == 6:
        print("Breaking at 6!")
        break
    print(i)

# Practical example: Search and stop
students = ["Alice", "Bob", "Charlie", "Diana", "Eve"]
target = "Charlie"
for student in students:
    if student == target:
        print(f"Found {target}!")
        break
else:
    print(f"{target} not found")

# TODO: Find first even number greater than 50


print("\n=== CONTINUE STATEMENT REVIEW ===\n")

# Continue skips the rest of current iteration
for i in range(1, 11):
    if i % 2 == 0:
        continue
    print(i)  # Only prints odd numbers

# Skip specific values
for i in range(1, 21):
    if i % 5 == 0:
        continue
    print(i, end=" ")
print()

# TODO: Print numbers 1-30, skipping multiples of 3 and 5


print("\n=== PASS STATEMENT ===\n")

# Pass does nothing - placeholder for future code
for i in range(5):
    if i == 2:
        pass  # TODO: Add logic here later
    print(i)

# Useful in development
for item in ["a", "b", "c"]:
    if item == "b":
        pass  # Will implement this later
    else:
        print(item)

# TODO: Create a loop structure you'll complete later using pass


print("\n=== COMBINING LOOPS AND CONDITIONS ===\n")

# Check multiple conditions in loop
numbers = [15, 22, 8, 35, 40, 3, 18]
for num in numbers:
    if num > 10 and num < 20:
        print(f"{num} is between 10 and 20")

# Multiple conditions with continue
for num in range(1, 21):
    if num % 2 == 0 or num % 3 == 0:
        continue
    print(num)  # Numbers not divisible by 2 or 3

# TODO: Print numbers 1-50 that are divisible by 7 but not by 2


print("\n=== NESTED LOOPS WITH CONTROL ===\n")

# Break out of inner loop only
for i in range(1, 4):
    for j in range(1, 6):
        if j == 3:
            break  # Only breaks inner loop
        print(f"({i},{j})", end=" ")
    print()

# Using flags to break outer loop
found = False
for i in range(1, 6):
    for j in range(1, 6):
        if i * j == 12:
            print(f"Found: {i} * {j} = 12")
            found = True
            break
    if found:
        break

# TODO: Find the first pair of numbers (1-20) whose product is exactly 100


print("\n=== LOOP PATTERNS ===\n")

# Pattern 1: Filtering
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_numbers = []
for num in numbers:
    if num % 2 == 0:
        even_numbers.append(num)
print(f"Even numbers: {even_numbers}")

# Pattern 2: Transforming
numbers = [1, 2, 3, 4, 5]
doubled = []
for num in numbers:
    doubled.append(num * 2)
print(f"Doubled: {doubled}")

# Pattern 3: Aggregating
numbers = [10, 20, 30, 40, 50]
total = 0
count = 0
for num in numbers:
    total += num
    count += 1
average = total / count
print(f"Average: {average}")

# TODO: From list [3, 7, 2, 9, 1, 5, 8], get all numbers > 5


print("\n=== LOOP WITH MULTIPLE ITERATORS ===\n")

# Loop through two lists together
names = ["Alice", "Bob", "Charlie"]
scores = [85, 92, 78]

for i in range(len(names)):
    print(f"{names[i]}: {scores[i]}")

# Better way: zip()
for name, score in zip(names, scores):
    print(f"{name}: {score}")

# TODO: Combine lists of fruits and prices, print fruit: $price


print("\n=== ADVANCED LOOP CONTROL ===\n")

# Early exit with condition
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for num in numbers:
    if num > 100:  # Impossible with our list
        break
    if num % 2 == 0:
        continue
    print(num)
else:
    print("Checked all numbers")

# Counting with conditions
count = 0
for i in range(1, 101):
    if i % 3 == 0 and i % 5 == 0:
        count += 1
print(f"Numbers divisible by both 3 and 5: {count}")

# TODO: Count how many numbers 1-100 are divisible by 7 OR 11


print("\n=== PRACTICE EXERCISES ===\n")

# Exercise 1: Perfect Numbers
print("--- Find Perfect Numbers ---")
# A perfect number equals the sum of its divisors (excluding itself)
# Example: 6 = 1 + 2 + 3
# TODO: Find all perfect numbers between 1 and 30


# Exercise 2: Prime Factorization
print("\n--- Prime Factorization ---")
number = 60
# TODO: Find all prime factors
# Expected: 2, 2, 3, 5 (because 2*2*3*5 = 60)


# Exercise 3: List Processing
print("\n--- List Processing ---")
numbers = [12, 45, 23, 78, 34, 89, 56, 23, 67, 45]
# TODO:
# - Remove duplicates (create new list without duplicates)
# - Find second largest number
# - Count numbers greater than average


# Exercise 4: String Validation
print("\n--- Validate Input ---")
# TODO: Keep asking for input until user enters:
# - A string with at least 5 characters
# - Contains at least one letter
# - Contains at least one number


# Exercise 5: Matrix Sum
print("\n--- Matrix Operations ---")
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
# TODO:
# - Calculate sum of all elements
# - Find maximum element
# - Calculate sum of diagonal (1, 5, 9)


# Exercise 6: Number Pattern
print("\n--- Number Pattern ---")
# TODO: Print pattern:
# 1
# 2 3
# 4 5 6
# 7 8 9 10


# Exercise 7: Armstrong Number
print("\n--- Armstrong Numbers ---")
# An Armstrong number: sum of cubes of digits equals the number
# Example: 153 = 1³ + 5³ + 3³
# TODO: Find all Armstrong numbers between 100 and 999


# Exercise 8: Bubble Sort (Preview)
print("\n--- Simple Sorting ---")
numbers = [64, 34, 25, 12, 22, 11, 90]
# TODO: Sort the list using nested loops (bubble sort)
# Hint: Compare adjacent elements and swap if needed


print("\n=== BONUS CHALLENGES ===\n")

# Challenge 1: Longest Word
sentence = "The quick brown fox jumps over the lazy dog"
# TODO: Find the longest word and its length


# Challenge 2: Palindrome Words
words = ["racecar", "hello", "level", "world", "deed", "python"]
# TODO: Find and print all palindromes


# Challenge 3: Number Spiral
# TODO: Print a number spiral (5x5):
#  1  2  3  4  5
# 16 17 18 19  6
# 15 24 25 20  7
# 14 23 22 21  8
# 13 12 11 10  9
# (This is advanced - save for later if too difficult)


# Challenge 4: Common Elements
list1 = [1, 2, 3, 4, 5, 6]
list2 = [4, 5, 6, 7, 8, 9]
# TODO: Find elements that appear in both lists


print("\nExcellent! Next: 10_string_formatting.py")
