"""
Exercise 4: Input and Output

Learning Objectives:
- Get user input with input()
- Display output with print()
- Format strings with f-strings
- Type conversion with user input
"""

print("=== USER INPUT ===\n")

# The input() function always returns a string
name = input("What is your name? ")
print("Hello, " + name + "!")

# TODO: Ask the user for their favorite color and print a response


# Getting numeric input requires type conversion
print("\n--- Numeric Input ---")
age_string = input("How old are you? ")
age = int(age_string)  # Convert string to integer
print(f"In 10 years, you will be {age + 10} years old.")

# Shorter version (convert directly)
# age = int(input("How old are you? "))

# TODO: Ask for the user's height in centimeters
# Convert to float and calculate height in meters (divide by 100)


print("\n=== OUTPUT FORMATTING ===\n")

# Multiple ways to format output

# 1. Concatenation (only works with strings)
firstname = "Alice"
print("Hello " + firstname + "!")

# 2. Comma separation (automatically adds spaces)
age = 25
print("Name:", firstname, "Age:", age)

# 3. F-strings (recommended - most readable)
city = "Paris"
print(f"My name is {firstname}, I'm {age} years old, and I live in {city}.")

# F-strings can include expressions
price = 29.99
quantity = 3
print(f"Total cost: ${price * quantity}")

# TODO: Create variables for a product name, price, and quantity
# Print a formatted receipt using an f-string


# Number formatting with f-strings
print("\n--- Number Formatting ---")

pi = 3.14159265359
print(f"Pi to 2 decimals: {pi:.2f}")
print(f"Pi to 4 decimals: {pi:.4f}")

large_number = 1000000
print(f"Formatted with commas: {large_number:,}")

percentage = 0.85
print(f"Percentage: {percentage:.1%}")

# TODO: Format and print 1/3 to exactly 5 decimal places


# Alignment and width
print("\n--- Alignment ---")
print(f"{'Left':<10}|")    # Left aligned, width 10
print(f"{'Center':^10}|")  # Center aligned, width 10
print(f"{'Right':>10}|")   # Right aligned, width 10

# Useful for creating tables
print("\n--- Table Formatting ---")
print(f"{'Name':<15} {'Age':>5} {'City':<10}")
print(f"{'Alice':<15} {25:>5} {'Paris':<10}")
print(f"{'Bob':<15} {30:>5} {'London':<10}")
print(f"{'Charlie':<15} {35:>5} {'Tokyo':<10}")

# TODO: Add one more person to the table


print("\n=== PRACTICE EXERCISES ===\n")

# Exercise 1: Personal Information Form
print("--- Personal Information ---")
# TODO: Ask for the following and store in variables:
# - First name
# - Last name
# - Age
# - City
# Then print: "[First] [Last], age [Age], lives in [City]"


# Exercise 2: Simple Calculator
print("\n--- Calculator ---")
# TODO: Ask user for two numbers and an operation (+, -, *, /)
# Calculate and display the result
# Hint: You'll need if statements (we'll learn these next), so for now
# just ask for two numbers and show their sum


# Exercise 3: Shopping Cart
print("\n--- Shopping Cart ---")
# TODO: Ask for item name, price per unit, and quantity
# Calculate total cost and display formatted receipt
# Example output:
# Item: Apple
# Price: $2.50
# Quantity: 4
# Total: $10.00


# Exercise 4: Temperature Converter (Interactive)
print("\n--- Temperature Converter ---")
# TODO: Ask user for temperature in Fahrenheit
# Convert to Celsius and display both temperatures formatted to 1 decimal
# Formula: C = (F - 32) * 5/9


# Exercise 5: Age Calculator
print("\n--- Age Calculator ---")
# TODO: Ask for birth year
# Calculate age in current year (2024)
# Display: "You are [age] years old"
# Also calculate how many days old they are (approximate: age * 365)


print("\n=== BONUS CHALLENGES ===\n")

# Challenge 1: BMI Calculator
# TODO: Ask for weight (kg) and height (meters)
# Calculate BMI = weight / (height ** 2)
# Display result formatted to 2 decimals


# Challenge 2: Currency Converter
# TODO: Ask for amount in USD
# Convert to EUR (use rate: 1 USD = 0.85 EUR)
# Display both amounts formatted with 2 decimals and currency symbols


# Challenge 3: Circle Calculator
# TODO: Ask for radius
# Calculate and display:
# - Circumference = 2 * π * r
# - Area = π * r²
# Use 3.14159 for π


# Multiple inputs in one line
print("\n--- Bonus: Multiple Inputs ---")
# You can get multiple inputs at once (but it's less user-friendly)
# example: name, age = input("Enter name and age: ").split()
# The user would type: Alice 25

print("\nGreat job with input and output! Next: 05_string_basics.py")
