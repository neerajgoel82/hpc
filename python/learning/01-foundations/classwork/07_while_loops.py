"""
Exercise 7: While Loops

Learning Objectives:
- While loop syntax
- Loop conditions
- Infinite loops and how to avoid them
- Break and continue statements
- Common while loop patterns
"""

print("=== BASIC WHILE LOOP ===\n")

# While loop: Repeats as long as condition is True
count = 1
while count <= 5:
    print(f"Count: {count}")
    count += 1  # IMPORTANT: Must change the variable to avoid infinite loop

print("Loop finished!\n")

# Countdown example
number = 5
while number > 0:
    print(number)
    number -= 1
print("Blast off!")

# TODO: Create a while loop that prints even numbers from 2 to 10


print("\n=== USER INPUT WITH WHILE LOOP ===\n")

# Keep asking until valid input
# password = ""
# while password != "secret":
#     password = input("Enter password: ")
# print("Access granted!")

# A better pattern (avoid empty first value)
# while True:
#     password = input("Enter password: ")
#     if password == "secret":
#         break
# print("Access granted!")

# TODO: Write a loop that keeps asking for a number
# Stop when user enters a number greater than 10


print("\n=== ACCUMULATOR PATTERN ===\n")

# Common pattern: accumulate a sum
total = 0
count = 1
while count <= 5:
    total += count
    count += 1
print(f"Sum of 1 to 5: {total}")

# Calculate factorial
number = 5
factorial = 1
counter = 1
while counter <= number:
    factorial *= counter
    counter += 1
print(f"Factorial of {number}: {factorial}")

# TODO: Calculate the sum of all even numbers from 1 to 20


print("\n=== SENTINEL VALUE PATTERN ===\n")

# Stop when a specific value is entered
# print("Enter numbers to sum (0 to stop):")
# total = 0
# number = int(input("Enter number: "))
# while number != 0:
#     total += number
#     number = int(input("Enter number: "))
# print(f"Total: {total}")

# TODO: Modify the above to count how many numbers were entered


print("\n=== BREAK STATEMENT ===\n")

# Break: Exit the loop immediately
count = 1
while True:  # Infinite loop!
    print(count)
    count += 1
    if count > 5:
        break  # Exit the loop

# Finding a number
target = 7
number = 1
while True:
    if number == target:
        print(f"Found {target}!")
        break
    number += 1

# TODO: Use break to exit when user types "quit"
# Keep asking "Enter command: " and print what they entered


print("\n=== CONTINUE STATEMENT ===\n")

# Continue: Skip rest of current iteration, go to next
count = 0
while count < 10:
    count += 1
    if count % 2 == 0:  # If even
        continue  # Skip printing, go to next iteration
    print(count)  # Only prints odd numbers

# Another example: Skip multiples of 3
number = 0
while number < 15:
    number += 1
    if number % 3 == 0:
        continue
    print(number)

# TODO: Print numbers 1-20, but skip multiples of 5


print("\n=== WHILE-ELSE ===\n")

# Else clause executes when loop completes normally (not via break)
count = 1
while count <= 3:
    print(count)
    count += 1
else:
    print("Loop completed normally")

# With break (else won't execute)
count = 1
while count <= 10:
    if count == 5:
        print("Breaking at 5")
        break
    count += 1
else:
    print("This won't print because we used break")

# TODO: Search for a number in a range
# Use else to print "Not found" if loop completes without finding it


print("\n=== NESTED WHILE LOOPS ===\n")

# Loop inside another loop
row = 1
while row <= 3:
    col = 1
    while col <= 3:
        print(f"({row},{col})", end=" ")
        col += 1
    print()  # New line after each row
    row += 1

# TODO: Create a multiplication table (1-5) using nested while loops
# Output:
# 1  2  3  4  5
# 2  4  6  8  10
# 3  6  9  12 15
# etc.


print("\n=== COMMON PITFALLS ===\n")

# Pitfall 1: Infinite loop (forgetting to update counter)
# count = 1
# while count <= 5:
#     print(count)
#     # Forgot: count += 1
# This would loop forever!

# Pitfall 2: Off-by-one errors
count = 1
while count < 5:  # Will print 1,2,3,4 (not 5!)
    print(count)
    count += 1

# Should be: while count <= 5 to include 5

print("\n=== PRACTICE EXERCISES ===\n")

# Exercise 1: Number Guessing
print("--- Number Guessing ---")
secret = 7
# TODO: Let user keep guessing until they get it right
# Give hints: "Too high" or "Too low"
# Count number of attempts


# Exercise 2: Reverse a Number
print("\n--- Reverse Number ---")
number = 12345
reversed_num = 0
# TODO: Reverse the digits
# Hint: Use % to get last digit, // to remove it
# Expected output: 54321


# Exercise 3: Sum of Digits
print("\n--- Sum of Digits ---")
number = 12345
total = 0
# TODO: Calculate sum of all digits
# Expected: 1+2+3+4+5 = 15


# Exercise 4: Power Calculator
print("\n--- Power Calculator ---")
base = 2
exponent = 5
result = 1
# TODO: Calculate base^exponent without using **
# Use a while loop to multiply base by itself exponent times


# Exercise 5: Fibonacci Sequence
print("\n--- Fibonacci Sequence ---")
# TODO: Print first 10 Fibonacci numbers
# Fibonacci: each number is sum of previous two
# Start with: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34


# Exercise 6: Prime Number Checker
print("\n--- Prime Checker ---")
number = 17
# TODO: Check if number is prime using a while loop
# Prime: only divisible by 1 and itself


# Exercise 7: Password Validator
print("\n--- Password Validator ---")
# TODO: Keep asking for password until it meets requirements:
# - At least 8 characters
# - Contains at least one number
# - Contains at least one uppercase letter


# Exercise 8: Menu System
print("\n--- Simple Menu ---")
# TODO: Create a menu that keeps showing until user quits:
# 1. Say Hello
# 2. Show Date
# 3. Quit
# Perform action based on choice


print("\n=== BONUS CHALLENGES ===\n")

# Challenge 1: Collatz Conjecture
# Start with any positive integer
# If even: divide by 2
# If odd: multiply by 3 and add 1
# Repeat until you reach 1
# TODO: Implement and count steps for number 27


# Challenge 2: Digital Root
# Sum the digits of a number repeatedly until one digit remains
# Example: 38 → 3+8=11 → 1+1=2
# TODO: Find digital root of 9875


# Challenge 3: ATM Simulation
# TODO: Implement simple ATM:
# - Show balance
# - Deposit
# - Withdraw (check sufficient funds)
# - Exit
# Start with balance of $1000


print("\nExcellent! Next: 08_for_loops.py")
