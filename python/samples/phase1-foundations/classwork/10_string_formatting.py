"""
Exercise 10: String Formatting (Advanced)

Learning Objectives:
- F-strings in depth
- Format specifiers
- String templates
- Pretty printing data
"""

print("=== F-STRING BASICS REVIEW ===\n")

name = "Alice"
age = 25
print(f"My name is {name} and I'm {age} years old.")

# Expressions in f-strings
a, b = 10, 20
print(f"Sum: {a + b}, Product: {a * b}")

# TODO: Create f-string with calculation of area (length * width)


print("\n=== NUMBER FORMATTING ===\n")

# Decimal places
pi = 3.14159265359
print(f"Pi: {pi:.2f}")  # 2 decimal places
print(f"Pi: {pi:.4f}")  # 4 decimal places
print(f"Pi: {pi:.6f}")  # 6 decimal places

# Thousands separator
population = 1234567890
print(f"Population: {population:,}")
print(f"Population: {population:_}")  # Using underscore

# Percentages
ratio = 0.875
print(f"Success rate: {ratio:.1%}")
print(f"Success rate: {ratio:.2%}")

# TODO: Format 1/3 to 3 decimal places and as percentage


print("\n=== WIDTH AND ALIGNMENT ===\n")

# Width (minimum space)
num = 42
print(f"Number: {num:5}")  # Minimum width 5

# Alignment
print(f"{'Left':<10}|")     # Left align
print(f"{'Center':^10}|")   # Center align
print(f"{'Right':>10}|")    # Right align

# With fill character
print(f"{'Title':=^20}")
print(f"{'Price':*>15}")

# Combining width and decimals
price = 19.99
print(f"Price: ${price:>8.2f}")

# TODO: Create a formatted table of items and prices


print("\n=== NUMBER SYSTEMS ===\n")

number = 255

# Binary, Octal, Hexadecimal
print(f"Decimal: {number}")
print(f"Binary: {number:b}")
print(f"Octal: {number:o}")
print(f"Hexadecimal: {number:x}")
print(f"Hexadecimal (upper): {number:X}")

# With prefix
print(f"Binary: {number:#b}")
print(f"Octal: {number:#o}")
print(f"Hex: {number:#x}")

# TODO: Display number 100 in all formats with prefixes


print("\n=== SIGN AND ZERO PADDING ===\n")

# Sign
positive = 42
negative = -42
print(f"Default: {positive:+}  {negative:+}")
print(f"Space for positive: {positive: }  {negative: }")

# Zero padding
num = 42
print(f"Zero padded: {num:05}")  # 00042
print(f"Zero padded: {num:08}")  # 00000042

# Useful for IDs
user_id = 7
print(f"User ID: {user_id:04d}")  # 0007

# TODO: Format invoice numbers like: INV-0001, INV-0123


print("\n=== DATES AND TIMES ===\n")

from datetime import datetime

now = datetime.now()
print(f"Full: {now}")
print(f"Date: {now:%Y-%m-%d}")
print(f"Time: {now:%H:%M:%S}")
print(f"Custom: {now:%B %d, %Y}")
print(f"12-hour: {now:%I:%M %p}")

# TODO: Print current date as "Day, Month DD, YYYY"
# Example: "Thursday, February 19, 2026"


print("\n=== CREATING TABLES ===\n")

# Simple table
print(f"{'Name':<15} {'Age':>5} {'City':<10}")
print("-" * 32)
print(f"{'Alice':<15} {25:>5} {'Paris':<10}")
print(f"{'Bob':<15} {30:>5} {'London':<10}")
print(f"{'Charlie':<15} {35:>5} {'Tokyo':<10}")

# With numbers
print(f"\n{'Product':<15} {'Price':>10} {'Qty':>5} {'Total':>10}")
print("=" * 42)
items = [
    ("Apple", 2.50, 4),
    ("Banana", 1.75, 6),
    ("Cherry", 5.00, 2)
]
for product, price, qty in items:
    total = price * qty
    print(f"{product:<15} ${price:>9.2f} {qty:>5} ${total:>9.2f}")

# TODO: Create a grade report table


print("\n=== MULTILINE F-STRINGS ===\n")

name = "Alice"
age = 25
city = "Paris"
report = f"""
{'='*40}
          PERSONAL REPORT
{'='*40}
Name:     {name}
Age:      {age}
City:     {city}
Status:   {'Adult' if age >= 18 else 'Minor'}
{'='*40}
"""
print(report)

# TODO: Create a receipt template


print("\n=== NESTED F-STRINGS ===\n")

# F-string inside f-string
value = 42
width = 10
print(f"{value:>{width}}")

# Conditional formatting
score = 85
print(f"Grade: {score} - {('Excellent' if score >= 90 else 'Good' if score >= 75 else 'Pass' if score >= 60 else 'Fail')}")

# TODO: Format temperature with color word (Hot/Warm/Cold) based on value


print("\n=== DEBUG FORMATTING (Python 3.8+) ===\n")

# The = specifier for debugging
x = 10
y = 20
print(f"{x = }")
print(f"{y = }")
print(f"{x + y = }")

# TODO: Use = to debug a calculation


print("\n=== PRACTICE EXERCISES ===\n")

# Exercise 1: Invoice Generator
print("--- Invoice ---")
items = [
    {"name": "Laptop", "price": 999.99, "qty": 1},
    {"name": "Mouse", "price": 29.99, "qty": 2},
    {"name": "Keyboard", "price": 79.99, "qty": 1}
]
# TODO: Create a formatted invoice with:
# - Header
# - Item rows with calculations
# - Subtotal, tax (10%), total


# Exercise 2: Grade Report
print("\n--- Grade Report ---")
students = [
    {"name": "Alice", "math": 85, "english": 92, "science": 88},
    {"name": "Bob", "math": 78, "english": 85, "science": 80},
    {"name": "Charlie", "math": 95, "english": 88, "science": 92}
]
# TODO: Print formatted report with:
# - Name, scores, average
# - Calculate overall class average


# Exercise 3: Progress Bar
print("\n--- Progress Bar ---")
total = 100
completed = 67
# TODO: Create text progress bar
# [===================>        ] 67%


# Exercise 4: Calendar Display
print("\n--- Mini Calendar ---")
# TODO: Print current month in calendar format
# Use datetime and formatting


# Exercise 5: Scoreboard
print("\n--- Scoreboard ---")
scores = [
    {"rank": 1, "player": "Alice", "score": 15000, "time": "05:23"},
    {"rank": 2, "player": "Bob", "score": 12500, "time": "06:15"},
    {"rank": 3, "player": "Charlie", "score": 10000, "time": "07:45"}
]
# TODO: Format as a nice scoreboard


print("\n=== BONUS CHALLENGES ===\n")

# Challenge 1: ASCII Art with Data
# TODO: Create ASCII art box with dynamic data inside


# Challenge 2: Colorful Output (using ANSI codes)
# ANSI codes for colors (works in most terminals)
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

print(f"{RED}Error message{RESET}")
print(f"{GREEN}Success message{RESET}")
# TODO: Use colors in a formatted report


# Challenge 3: Dynamic Column Widths
data = [
    ["Name", "Age", "Occupation"],
    ["Alice", "25", "Engineer"],
    ["Bob", "30", "Designer"],
    ["Christopher", "35", "Manager"]
]
# TODO: Calculate optimal column widths based on data


print("\nGreat job! Complete the Week 2 project: project_guessing_game.py")
