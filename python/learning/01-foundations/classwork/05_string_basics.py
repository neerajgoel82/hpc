"""
Exercise 5: String Basics

Learning Objectives:
- String creation and indexing
- String methods
- String slicing
- Common string operations
"""

print("=== STRING CREATION ===\n")

# Three ways to create strings
single_quotes = 'Hello'
double_quotes = "World"
triple_quotes = """This is a
multi-line string"""

print(single_quotes)
print(double_quotes)
print(triple_quotes)

# When to use what?
# - Single or double quotes: personal preference (be consistent)
# - Triple quotes: multi-line strings
# - Use one type to include the other: "It's a sunny day" or 'He said "hello"'

# Escape characters
message = "She said, \"Hello!\""
path = "C:\\Users\\Documents"  # Use \\ for backslash
new_line = "Line 1\nLine 2\nLine 3"

print(message)
print(path)
print(new_line)

# TODO: Create a string with a tab character (\t) between words


print("\n=== STRING INDEXING ===\n")

text = "Python"
#       012345  (positive indices)
#      -6-5-4-3-2-1  (negative indices)

print(f"First character: {text[0]}")  # P
print(f"Third character: {text[2]}")  # t
print(f"Last character: {text[-1]}")  # n
print(f"Second to last: {text[-2]}")  # o

# TODO: Get and print the first and last letter of your name


print("\n=== STRING SLICING ===\n")

phrase = "Hello World"
#         01234567891011

# Slicing: string[start:end]  (end is not included)
print(f"First 5 characters: {phrase[0:5]}")  # Hello
print(f"Characters 6-10: {phrase[6:11]}")    # World
print(f"First 5 (shorthand): {phrase[:5]}")  # Hello
print(f"From index 6 to end: {phrase[6:]}")  # World
print(f"Last 5 characters: {phrase[-5:]}")   # World

# Step parameter: string[start:end:step]
numbers = "0123456789"
print(f"Every 2nd character: {numbers[::2]}")   # 02468
print(f"Reverse string: {numbers[::-1]}")       # 9876543210

# TODO: From the string "Programming", extract:
# - First 7 characters
# - Last 4 characters
# - Every 2nd character
# - The string in reverse


print("\n=== STRING METHODS ===\n")

sample = "  Hello, Python World!  "

# Case conversion
print(f"Upper: {sample.upper()}")
print(f"Lower: {sample.lower()}")
print(f"Title: {sample.title()}")
print(f"Capitalize: {sample.capitalize()}")

# Whitespace removal
print(f"Strip (both ends): '{sample.strip()}'")
print(f"Lstrip (left): '{sample.lstrip()}'")
print(f"Rstrip (right): '{sample.rstrip()}'")

# Find and replace
email = "user@example.com"
print(f"Replace: {email.replace('example', 'company')}")

sentence = "I love Java"
print(f"Replace: {sentence.replace('Java', 'Python')}")

# Finding substrings
text = "Python is awesome"
print(f"Index of 'is': {text.find('is')}")  # Returns -1 if not found
print(f"Index of 'xyz': {text.find('xyz')}")
print(f"Count 'o': {text.count('o')}")

# Checking string content
print(f"Starts with 'Py': {text.startswith('Py')}")
print(f"Ends with 'some': {text.endswith('some')}")

# TODO: Take the string "   HELLO python   "
# - Remove whitespace
# - Convert to title case
# - Replace "python" with "World"


print("\n=== STRING CHECKS ===\n")

# Checking string types
alpha = "Hello"
numeric = "12345"
alphanumeric = "Hello123"
spaces = "   "

print(f"'{alpha}'.isalpha(): {alpha.isalpha()}")  # Letters only
print(f"'{numeric}'.isdigit(): {numeric.isdigit()}")  # Digits only
print(f"'{alphanumeric}'.isalnum(): {alphanumeric.isalnum()}")  # Letters & digits
print(f"'{spaces}'.isspace(): {spaces.isspace()}")  # Whitespace only

# TODO: Check if "Python3" is alphanumeric


print("\n=== STRING CONCATENATION & REPETITION ===\n")

first = "Hello"
second = "World"

# Concatenation
combined = first + " " + second
print(combined)

# Repetition
laugh = "ha" * 5
print(laugh)  # hahahahaha

border = "=" * 40
print(border)

# TODO: Create a string "Python" and repeat it 3 times with spaces


print("\n=== STRING SPLITTING & JOINING ===\n")

# Split string into list
sentence = "Python is easy to learn"
words = sentence.split()  # Splits by whitespace
print(f"Words: {words}")

csv_data = "apple,banana,cherry"
fruits = csv_data.split(",")
print(f"Fruits: {fruits}")

# Join list into string
word_list = ["I", "love", "Python"]
joined = " ".join(word_list)
print(f"Joined: {joined}")

# TODO: Take "one-two-three-four" and split by "-"


print("\n=== PRACTICE EXERCISES ===\n")

# Exercise 1: Email Validator
print("--- Email Validator ---")
email = "user@example.com"
# TODO: Check if email contains "@" and "."
# Print True if valid format, False otherwise


# Exercise 2: Username Generator
print("\n--- Username Generator ---")
first_name = "John"
last_name = "Doe"
birth_year = 1995
# TODO: Create username as: first 3 letters of first name +
# first 3 letters of last name + last 2 digits of year
# Example: "JohDoe95"
# Convert to lowercase


# Exercise 3: Password Strength Checker
print("\n--- Password Strength ---")
password = "MyPass123"
# TODO: Check the following and print results:
# - Length is at least 8 characters
# - Contains at least one digit
# - Contains at least one letter


# Exercise 4: String Cleaner
print("\n--- String Cleaner ---")
messy_text = "   PyThOn   Is   AwEsOmE!!!   "
# TODO:
# - Remove extra spaces from ends
# - Convert to lowercase
# - Remove exclamation marks
# - Print cleaned text


# Exercise 5: Acronym Generator
print("\n--- Acronym Generator ---")
phrase = "North Atlantic Treaty Organization"
# TODO: Create acronym by taking first letter of each word
# Expected output: "NATO"
# Hint: Split the phrase and use a loop (or do it manually for now)


# Exercise 6: Palindrome Checker
print("\n--- Palindrome Checker ---")
word = "racecar"
# TODO: Check if word is same forwards and backwards
# Hint: Compare word with its reverse


# Exercise 7: Word Counter
print("\n--- Word Counter ---")
text = "Python is fun. Python is powerful. I love Python!"
# TODO: Count how many times "Python" appears in the text


# Exercise 8: String Reverser
print("\n--- String Reverser ---")
original = "Hello World"
# TODO: Reverse the string and print it


print("\n=== BONUS: STRING FORMATTING TRICKS ===\n")

# Center text
title = "WELCOME"
print(title.center(40, "="))

# Create a simple box
message = "Python"
print("+" + "-" * 20 + "+")
print("|" + message.center(20) + "|")
print("+" + "-" * 20 + "+")

# TODO: Create a box around your name


print("\nExcellent! You've mastered string basics!")
print("Complete the Week 1 project: project_calculator.py")
