"""
Exercise 8: For Loops

Learning Objectives:
- For loop syntax
- Range function
- Iterating over strings, lists
- Break and continue with for loops
- Nested for loops
"""

print("=== BASIC FOR LOOP ===\n")

# For loop: Iterate over a sequence
for i in range(5):
    print(f"Number: {i}")

# Note: range(5) gives 0, 1, 2, 3, 4 (doesn't include 5)

# Range with start and end
for i in range(1, 6):  # 1 to 5
    print(i)

# Range with step
for i in range(0, 10, 2):  # Even numbers 0 to 8
    print(i)

# TODO: Print odd numbers from 1 to 20 using range


print("\n=== ITERATING OVER STRINGS ===\n")

# Loop through each character
message = "Python"
for char in message:
    print(char)

# With index using enumerate
for index, char in enumerate(message):
    print(f"Index {index}: {char}")

# TODO: Count how many vowels are in "Programming"


print("\n=== ITERATING OVER LISTS ===\n")

fruits = ["apple", "banana", "cherry"]

# Basic iteration
for fruit in fruits:
    print(f"I like {fruit}")

# With index
for i in range(len(fruits)):
    print(f"{i}: {fruits[i]}")

# Better way: enumerate
for index, fruit in enumerate(fruits):
    print(f"{index}: {fruit}")

# With start index
for index, fruit in enumerate(fruits, start=1):
    print(f"{index}. {fruit}")

# TODO: Create a list of 5 colors and print each with its position


print("\n=== RANGE() IN DETAIL ===\n")

# range(stop) - from 0 to stop-1
print("range(5):", list(range(5)))

# range(start, stop) - from start to stop-1
print("range(2, 7):", list(range(2, 7)))

# range(start, stop, step)
print("range(0, 10, 2):", list(range(0, 10, 2)))

# Backwards
print("range(10, 0, -1):", list(range(10, 0, -1)))

# TODO: Create a countdown from 10 to 1 using range


print("\n=== ACCUMULATOR PATTERN ===\n")

# Sum numbers 1 to 10
total = 0
for i in range(1, 11):
    total += i
print(f"Sum of 1 to 10: {total}")

# Calculate factorial
n = 5
factorial = 1
for i in range(1, n + 1):
    factorial *= i
print(f"Factorial of {n}: {factorial}")

# Build a string
result = ""
for i in range(1, 6):
    result += str(i)
print(f"Built string: {result}")

# TODO: Calculate the sum of squares of numbers 1 to 10
# 1^2 + 2^2 + 3^2 + ... + 10^2


print("\n=== BREAK AND CONTINUE ===\n")

# Break: Exit loop early
for i in range(10):
    if i == 5:
        print("Breaking at 5")
        break
    print(i)

# Continue: Skip to next iteration
for i in range(10):
    if i % 2 == 0:
        continue  # Skip even numbers
    print(i)

# Find first number divisible by 7
for i in range(1, 100):
    if i % 7 == 0:
        print(f"First number divisible by 7: {i}")
        break

# TODO: Find the first number divisible by both 3 and 5


print("\n=== NESTED FOR LOOPS ===\n")

# Loop inside a loop
for i in range(1, 4):
    for j in range(1, 4):
        print(f"({i},{j})", end=" ")
    print()  # New line after inner loop

# Multiplication table
print("\nMultiplication Table:")
for i in range(1, 6):
    for j in range(1, 6):
        print(f"{i*j:3}", end=" ")
    print()

# TODO: Print a right triangle pattern:
# *
# **
# ***
# ****
# *****


print("\n=== FOR-ELSE ===\n")

# Else executes if loop completes without break
for i in range(5):
    print(i)
else:
    print("Loop completed normally")

# With break (else won't execute)
for i in range(10):
    if i == 5:
        break
else:
    print("This won't print")

# Practical use: searching
numbers = [1, 3, 5, 7, 9]
target = 6
for num in numbers:
    if num == target:
        print(f"Found {target}")
        break
else:
    print(f"{target} not found")

# TODO: Search for the letter 'x' in "Python Programming"


print("\n=== COMMON PATTERNS ===\n")

# Pattern 1: Counting
count = 0
for i in range(1, 21):
    if i % 3 == 0:
        count += 1
print(f"Numbers divisible by 3: {count}")

# Pattern 2: Finding maximum
numbers = [45, 23, 67, 89, 12, 34]
maximum = numbers[0]
for num in numbers:
    if num > maximum:
        maximum = num
print(f"Maximum: {maximum}")

# Pattern 3: Building a list
squares = []
for i in range(1, 11):
    squares.append(i ** 2)
print(f"Squares: {squares}")

# TODO: Find the minimum value in [78, 45, 92, 23, 67]


print("\n=== PRACTICE EXERCISES ===\n")

# Exercise 1: Sum of Even Numbers
print("--- Sum of Even Numbers ---")
# TODO: Calculate sum of all even numbers from 1 to 100


# Exercise 2: Character Counter
print("\n--- Character Counter ---")
text = "Hello World"
# TODO: Count how many times each character appears
# (You'll need a dictionary - or use multiple if statements for now)


# Exercise 3: Factorial
print("\n--- Factorial ---")
# TODO: Calculate factorial of 10 (10!)


# Exercise 4: Prime Numbers
print("\n--- Prime Numbers ---")
# TODO: Print all prime numbers between 1 and 50


# Exercise 5: Reverse a String
print("\n--- Reverse String ---")
text = "Python"
# TODO: Reverse the string using a for loop


# Exercise 6: Pattern Printing
print("\n--- Patterns ---")
# TODO: Print the following patterns:
#
# Pattern 1:
# 1
# 12
# 123
# 1234
# 12345
#
# Pattern 2:
# *****
# ****
# ***
# **
# *
#
# Pattern 3 (pyramid):
#     *
#    ***
#   *****
#  *******
# *********


# Exercise 7: Fibonacci Sequence
print("\n--- Fibonacci ---")
# TODO: Print first 15 Fibonacci numbers


# Exercise 8: Temperature Converter Table
print("\n--- Temperature Table ---")
# TODO: Create table showing Celsius to Fahrenheit conversion
# From 0°C to 100°C in steps of 10
# Formula: F = C * 9/5 + 32


print("\n=== BONUS CHALLENGES ===\n")

# Challenge 1: Password Generator
# TODO: Generate a random password with:
# - 8 characters
# - Mix of letters and numbers
# Hint: Use chr() and range for ASCII values


# Challenge 2: Number Pyramid
# TODO: Print:
#     1
#    121
#   12321
#  1234321
# 123454321


# Challenge 3: FizzBuzz
# TODO: Print numbers 1 to 100, but:
# - For multiples of 3: print "Fizz"
# - For multiples of 5: print "Buzz"
# - For multiples of both: print "FizzBuzz"


# Challenge 4: List Statistics
numbers = [23, 45, 67, 12, 89, 34, 56, 78, 90, 11]
# TODO: Calculate and print:
# - Sum
# - Average
# - Maximum
# - Minimum
# - Range (max - min)


print("\nAwesome work! Next: 09_loop_control.py")
