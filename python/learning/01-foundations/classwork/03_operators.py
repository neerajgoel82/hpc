"""
Exercise 3: Operators

Learning Objectives:
- Arithmetic operators (+, -, *, /, //, %, **)
- Comparison operators (==, !=, <, >, <=, >=)
- Logical operators (and, or, not)
- Assignment operators (=, +=, -=, etc.)
"""

print("=== ARITHMETIC OPERATORS ===\n")

# Basic arithmetic
a = 10
b = 3

print(f"a = {a}, b = {b}")
print(f"Addition: {a} + {b} = {a + b}")
print(f"Subtraction: {a} - {b} = {a - b}")
print(f"Multiplication: {a} * {b} = {a * b}")
print(f"Division: {a} / {b} = {a / b}")  # Always returns float
print(f"Floor Division: {a} // {b} = {a // b}")  # Rounds down to nearest integer
print(f"Modulus (remainder): {a} % {b} = {a % b}")
print(f"Exponentiation: {a} ** {b} = {a ** b}")  # 10^3 = 1000

# TODO: Calculate the following and print results:
# - 25 divided by 4 (regular division)
# - 25 divided by 4 (floor division)
# - Remainder of 25 divided by 4
# - 2 to the power of 8


print("\n=== COMPARISON OPERATORS ===\n")

x = 5
y = 10
z = 5

print(f"x = {x}, y = {y}, z = {z}")
print(f"x == y: {x == y}")  # Equal to
print(f"x != y: {x != y}")  # Not equal to
print(f"x < y: {x < y}")    # Less than
print(f"x > y: {x > y}")    # Greater than
print(f"x <= z: {x <= z}")  # Less than or equal to
print(f"x >= z: {x >= z}")  # Greater than or equal to
print(f"x == z: {x == z}")  # Equal to

# TODO: Create three variables representing test scores (0-100)
# Compare them and print which score is the highest


print("\n=== LOGICAL OPERATORS ===\n")

# and: Both conditions must be True
# or: At least one condition must be True
# not: Reverses the boolean value

age = 20
has_license = True
has_car = False

print(f"Age: {age}, Has license: {has_license}, Has car: {has_car}")
print(f"Can drive? (age >= 18 and has_license): {age >= 18 and has_license}")
print(f"Can travel? (has_license or has_car): {has_license or has_car}")
print(f"Not has_car: {not has_car}")

# Combining multiple conditions
is_weekend = True
is_raining = False
print(f"\nGo to park? (is_weekend and not is_raining): {is_weekend and not is_raining}")

# TODO: Create variables for temperature and weather condition
# Write a logical expression that checks if it's good weather for a picnic:
# - Temperature between 20 and 30 degrees
# - Not raining
# - Is a weekend


print("\n=== ASSIGNMENT OPERATORS ===\n")

# Basic assignment
num = 10
print(f"num = {num}")

# Compound assignment operators (shorthand)
num += 5  # Same as: num = num + 5
print(f"After num += 5: {num}")

num -= 3  # Same as: num = num - 3
print(f"After num -= 3: {num}")

num *= 2  # Same as: num = num * 2
print(f"After num *= 2: {num}")

num //= 4  # Same as: num = num // 4
print(f"After num //= 4: {num}")

# TODO: Start with a variable score = 100
# - Add 50 points (bonus)
# - Subtract 25 points (penalty)
# - Multiply by 2 (double points event)
# - Print the final score


print("\n=== OPERATOR PRECEDENCE ===\n")

# Python follows mathematical order of operations (PEMDAS)
# Parentheses, Exponents, Multiplication/Division, Addition/Subtraction

result1 = 2 + 3 * 4
print(f"2 + 3 * 4 = {result1}")  # Multiplication first: 2 + 12 = 14

result2 = (2 + 3) * 4
print(f"(2 + 3) * 4 = {result2}")  # Parentheses first: 5 * 4 = 20

result3 = 10 + 5 * 2 ** 3 - 8 / 4
print(f"10 + 5 * 2 ** 3 - 8 / 4 = {result3}")

# TODO: Calculate the area of a circle with radius 5
# Formula: π * r²
# Use 3.14159 for π


print("\n=== PRACTICE CHALLENGES ===\n")

# Challenge 1: Temperature Converter
# Convert 77 degrees Fahrenheit to Celsius
# Formula: C = (F - 32) * 5/9
fahrenheit = 77
# TODO: Calculate celsius and print it


# Challenge 2: Even or Odd
# Check if a number is even (hint: use modulus operator)
number = 17
# TODO: Create a boolean variable is_even that's True if number is even
# Print the result


# Challenge 3: Discount Calculator
original_price = 120
discount_percent = 15
# TODO: Calculate the final price after discount
# Print both the discount amount and final price


# Challenge 4: Grade Checker
score = 85
passing_score = 60
perfect_score = 100
# TODO: Create boolean expressions for:
# - is_passing: score is >= passing_score
# - is_perfect: score is == perfect_score
# - needs_improvement: score is < 70
# Print all three


print("\nExcellent work! Next up: 04_input_output.py")
