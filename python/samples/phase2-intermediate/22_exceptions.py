"""
Exercise 22: Exception Handling

Learning Objectives:
- Try-except blocks
- Common exceptions
- Multiple except blocks
- Finally clause
- Raising exceptions
"""

print("=== BASIC TRY-EXCEPT ===\n")

# Without exception handling (would crash)
# number = int("abc")  # ValueError!

# With exception handling
try:
    number = int("abc")
    print(f"Number: {number}")
except ValueError:
    print("Error: Cannot convert to integer!")

print("Program continues...\n")

# Practical example: Safe division
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Error: Cannot divide by zero!")

# TODO: Handle conversion of invalid string to integer


print("\n=== MULTIPLE EXCEPT BLOCKS ===\n")

def safe_divide(a, b):
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        print("Error: Division by zero!")
        return None
    except TypeError:
        print("Error: Invalid types for division!")
        return None

print(safe_divide(10, 2))
print(safe_divide(10, 0))
print(safe_divide(10, "2"))

# TODO: Create function with multiple exception handlers


print("\n=== EXCEPT WITH VARIABLE ===\n")

try:
    number = int("abc")
except ValueError as e:
    print(f"ValueError occurred: {e}")

# Multiple exceptions, one handler
try:
    # result = 10 / 0
    number = int("abc")
except (ValueError, ZeroDivisionError) as e:
    print(f"Error: {e}")

# TODO: Catch and print exception details


print("\n=== ELSE CLAUSE ===\n")

# Else runs if no exception occurs
try:
    number = int("123")
except ValueError:
    print("Invalid number")
else:
    print(f"Successfully converted: {number}")

# Practical use
def process_file(filename):
    try:
        file = open(filename, 'r')
    except FileNotFoundError:
        print(f"File {filename} not found")
    else:
        content = file.read()
        file.close()
        print(f"File read successfully")

# TODO: Use try-except-else for user input validation


print("\n=== FINALLY CLAUSE ===\n")

# Finally always runs (cleanup code)
try:
    file = open("test.txt", 'r')
    content = file.read()
except FileNotFoundError:
    print("File not found")
finally:
    print("Cleanup code runs no matter what")
    # file.close()  # Would close if file was opened

# Example: Database connection
def connect_to_db():
    print("Connecting to database...")
    try:
        # connection = connect()
        print("Performing operations...")
        # raise Exception("Database error!")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Closing database connection")

connect_to_db()

# TODO: Use finally for cleanup


print("\n=== RAISING EXCEPTIONS ===\n")

def set_age(age):
    if age < 0:
        raise ValueError("Age cannot be negative!")
    if age > 150:
        raise ValueError("Age too high!")
    print(f"Age set to {age}")

set_age(25)
try:
    set_age(-5)
except ValueError as e:
    print(f"Error: {e}")

# TODO: Create function that raises exceptions for invalid input


print("\n=== COMMON EXCEPTIONS ===\n")

# ValueError: Invalid value
# int("abc")

# TypeError: Invalid type
# "2" + 2

# ZeroDivisionError: Division by zero
# 10 / 0

# IndexError: Invalid index
# [1, 2, 3][10]

# KeyError: Invalid key
# {"a": 1}["b"]

# FileNotFoundError: File doesn't exist
# open("nonexistent.txt")

# AttributeError: Invalid attribute
# "hello".nonexistent()

# TODO: Create examples of each exception type


print("\n=== PRACTICE EXERCISES ===\n")

# Exercise 1: Safe Input
# TODO: Get integer from user with exception handling


# Exercise 2: Safe List Access
# TODO: Function that safely accesses list element


# Exercise 3: File Reader
# TODO: Function that reads file with error handling


# Exercise 4: Calculator
# TODO: Create calculator with exception handling


# Exercise 5: Validation
# TODO: Validate email format (raise exceptions)


print("\nExcellent! Next: 23_file_reading.py")
