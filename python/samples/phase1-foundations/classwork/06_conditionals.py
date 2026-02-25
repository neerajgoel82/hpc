"""
Exercise 6: Conditionals (If/Elif/Else)

Learning Objectives:
- If statements
- Elif (else if) chains
- Else clauses
- Nested conditions
- Ternary operators
"""

print("=== BASIC IF STATEMENTS ===\n")

# If statement: Execute code only if condition is True
age = 20

if age >= 18:
    print("You are an adult")

# Note the indentation! Python uses indentation to define code blocks

temperature = 35
if temperature > 30:
    print("It's hot outside!")
    print("Drink plenty of water")

# TODO: Check if a variable 'score' is greater than 50
# If yes, print "You passed!"


print("\n=== IF-ELSE STATEMENTS ===\n")

# If-else: Execute one block or the other
age = 15

if age >= 18:
    print("You can vote")
else:
    print("You cannot vote yet")

# Another example
number = 7
if number % 2 == 0:
    print(f"{number} is even")
else:
    print(f"{number} is odd")

# TODO: Check if a number is positive or negative
# Print appropriate message


print("\n=== IF-ELIF-ELSE CHAINS ===\n")

# Use elif when you have multiple conditions to check
score = 85

if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
elif score >= 60:
    grade = "D"
else:
    grade = "F"

print(f"Score: {score}, Grade: {grade}")

# Traffic light example
light = "yellow"

if light == "red":
    print("Stop!")
elif light == "yellow":
    print("Slow down")
elif light == "green":
    print("Go!")
else:
    print("Invalid light color")

# TODO: Create a temperature classifier:
# - Above 30: "Hot"
# - 20-30: "Warm"
# - 10-20: "Cool"
# - Below 10: "Cold"


print("\n=== NESTED CONDITIONS ===\n")

# You can put if statements inside other if statements
age = 25
has_license = True

if age >= 18:
    if has_license:
        print("You can drive")
    else:
        print("You need a license to drive")
else:
    print("You're too young to drive")

# Another example
username = "admin"
password = "secret123"

if username == "admin":
    if password == "secret123":
        print("Login successful!")
    else:
        print("Wrong password")
else:
    print("User not found")

# TODO: Check if a person can watch an R-rated movie
# Requirements: age >= 17 OR (age >= 13 AND has_parent_permission)


print("\n=== MULTIPLE CONDITIONS ===\n")

# Using 'and' - both conditions must be True
age = 25
income = 30000

if age >= 18 and income >= 25000:
    print("You qualify for the loan")

# Using 'or' - at least one condition must be True
day = "Saturday"

if day == "Saturday" or day == "Sunday":
    print("It's the weekend!")

# Combining multiple logical operators
temperature = 25
is_raining = False

if temperature > 20 and temperature < 30 and not is_raining:
    print("Perfect weather for a picnic!")

# TODO: Check if a person can board a plane:
# - Has valid passport AND
# - Has boarding pass AND
# - (No luggage OR luggage weight <= 23kg)


print("\n=== TERNARY OPERATOR ===\n")

# Shorthand for simple if-else
age = 20
status = "adult" if age >= 18 else "minor"
print(status)

# Traditional way (same result):
# if age >= 18:
#     status = "adult"
# else:
#     status = "minor"

# Another example
score = 75
result = "Pass" if score >= 60 else "Fail"
print(f"Score: {score}, Result: {result}")

# TODO: Use ternary operator to set a variable 'size':
# "Large" if number > 100, otherwise "Small"


print("\n=== PRACTICE EXERCISES ===\n")

# Exercise 1: Age Classifier
print("--- Age Classifier ---")
age = 35
# TODO: Classify into: Child (0-12), Teen (13-19), Adult (20-64), Senior (65+)


# Exercise 2: Login System
print("\n--- Login System ---")
stored_username = "user123"
stored_password = "pass456"
input_username = "user123"
input_password = "pass456"
# TODO: Check if credentials match and print appropriate message


# Exercise 3: Discount Calculator
print("\n--- Discount Calculator ---")
purchase_amount = 150
is_member = True
# TODO: Calculate discount:
# - Members get 20% off on purchases over 100
# - Members get 10% off on purchases under 100
# - Non-members get 5% off on purchases over 100
# - No discount for non-members under 100


# Exercise 4: Leap Year Checker
print("\n--- Leap Year Checker ---")
year = 2024
# TODO: A year is a leap year if:
# - Divisible by 4 AND (not divisible by 100 OR divisible by 400)
# Examples: 2000 (yes), 1900 (no), 2024 (yes)


# Exercise 5: Grade with Plus/Minus
print("\n--- Grade with +/- ---")
score = 87
# TODO: Assign grade with + or -
# A: 90-100 (A-: 90-92, A: 93-97, A+: 98-100)
# B: 80-89 (B-: 80-82, B: 83-87, B+: 88-89)
# C: 70-79 (similar pattern)
# D: 60-69, F: below 60


# Exercise 6: Triangle Type Checker
print("\n--- Triangle Type ---")
side1, side2, side3 = 5, 5, 5
# TODO: Determine if triangle is:
# - Equilateral (all sides equal)
# - Isosceles (two sides equal)
# - Scalene (no sides equal)
# Also check if it's a valid triangle (sum of any two sides > third side)


# Exercise 7: BMI Category
print("\n--- BMI Category ---")
bmi = 22.5
# TODO: Classify BMI:
# Underweight: < 18.5
# Normal: 18.5 - 24.9
# Overweight: 25 - 29.9
# Obese: >= 30


# Exercise 8: Ticket Pricing
print("\n--- Ticket Pricing ---")
age = 25
is_student = False
is_senior = False
# TODO: Calculate ticket price:
# Base price: $15
# Children (under 12): 50% off
# Students: 20% off
# Seniors (65+): 30% off
# Free for children under 3


print("\n=== BONUS CHALLENGES ===\n")

# Challenge 1: Rock, Paper, Scissors
player1 = "rock"
player2 = "scissors"
# TODO: Determine winner


# Challenge 2: Time of Day Greeting
hour = 14  # 24-hour format
# TODO: Print greeting based on time:
# 5-11: "Good morning"
# 12-17: "Good afternoon"
# 18-21: "Good evening"
# 22-4: "Good night"


# Challenge 3: Password Strength Validator
password = "MyP@ss123"
# TODO: Check password strength (strong/medium/weak):
# Strong: length >= 12, has uppercase, lowercase, number, special char
# Medium: length >= 8, has uppercase, lowercase, and (number or special)
# Weak: everything else


print("\nGreat work! Next: 07_while_loops.py")
