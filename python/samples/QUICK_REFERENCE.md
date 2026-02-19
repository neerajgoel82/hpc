# Python Quick Reference

A cheat sheet for quick lookup while coding.

## Basic Syntax

### Print & Comments
```python
print("Hello")              # Print to console
# This is a comment
"""This is a
multi-line comment"""
```

### Variables
```python
name = "Alice"              # String
age = 25                    # Integer
price = 19.99               # Float
is_active = True            # Boolean
nothing = None              # None type
```

### Data Types
```python
type(42)                    # <class 'int'>
int("10")                   # Convert to int
float("3.14")               # Convert to float
str(100)                    # Convert to string
```

## Operators

### Arithmetic
```python
10 + 5      # Addition: 15
10 - 5      # Subtraction: 5
10 * 5      # Multiplication: 50
10 / 5      # Division: 2.0
10 // 3     # Floor division: 3
10 % 3      # Modulus: 1
10 ** 2     # Exponentiation: 100
```

### Comparison
```python
10 == 10    # Equal: True
10 != 5     # Not equal: True
10 > 5      # Greater than: True
10 < 5      # Less than: False
10 >= 10    # Greater or equal: True
10 <= 10    # Less or equal: True
```

### Logical
```python
True and False  # False
True or False   # True
not True        # False
```

## Strings

### Basic Operations
```python
s = "Hello"
s + " World"            # Concatenation: "Hello World"
s * 3                   # Repetition: "HelloHelloHello"
s[0]                    # Index: "H"
s[-1]                   # Last char: "o"
s[1:4]                  # Slice: "ell"
len(s)                  # Length: 5
```

### Methods
```python
s.upper()               # "HELLO"
s.lower()               # "hello"
s.strip()               # Remove whitespace
s.replace("l", "L")     # Replace chars
s.split()               # Split into list
s.startswith("He")      # True
s.endswith("lo")        # True
"x" in s                # Check membership: False
```

### Formatting
```python
name = "Alice"
age = 25
f"I'm {name}, {age}"    # f-string: "I'm Alice, 25"
f"{age:05d}"            # Zero pad: "00025"
f"{3.14159:.2f}"        # Decimals: "3.14"
```

## Lists

### Creation & Access
```python
lst = [1, 2, 3, 4, 5]
lst[0]                  # First: 1
lst[-1]                 # Last: 5
lst[1:3]                # Slice: [2, 3]
lst[::-1]               # Reverse: [5,4,3,2,1]
```

### Methods
```python
lst.append(6)           # Add to end
lst.insert(0, 0)        # Insert at position
lst.remove(3)           # Remove first 3
lst.pop()               # Remove & return last
lst.sort()              # Sort in place
sorted(lst)             # Return sorted copy
lst.reverse()           # Reverse in place
len(lst)                # Length
```

### List Comprehension
```python
[x**2 for x in range(5)]              # [0,1,4,9,16]
[x for x in range(10) if x % 2 == 0]  # [0,2,4,6,8]
```

## Dictionaries

### Creation & Access
```python
d = {"name": "Alice", "age": 25}
d["name"]               # Access value: "Alice"
d.get("city", "NYC")    # Get with default
d["city"] = "Paris"     # Add/update
```

### Methods
```python
d.keys()                # All keys
d.values()              # All values
d.items()               # Key-value pairs
del d["age"]            # Delete key
d.pop("name")           # Remove & return
"name" in d             # Check key exists
```

### Dict Comprehension
```python
{x: x**2 for x in range(5)}  # {0:0, 1:1, 2:4, 3:9, 4:16}
```

## Sets

### Creation & Operations
```python
s = {1, 2, 3}
s.add(4)                # Add element
s.remove(2)             # Remove element
s1 | s2                 # Union
s1 & s2                 # Intersection
s1 - s2                 # Difference
s1 ^ s2                 # Symmetric difference
```

## Tuples

### Immutable Lists
```python
t = (1, 2, 3)
t[0]                    # Access: 1
t.count(2)              # Count occurrences
t.index(3)              # Find index
x, y, z = t             # Unpacking
```

## Control Flow

### If Statements
```python
if condition:
    # do something
elif other_condition:
    # do something else
else:
    # default action

# Ternary operator
value = x if condition else y
```

### For Loops
```python
for i in range(5):      # 0,1,2,3,4
    print(i)

for item in list:       # Iterate list
    print(item)

for i, item in enumerate(list):  # With index
    print(i, item)

for k, v in dict.items():  # Iterate dict
    print(k, v)
```

### While Loops
```python
while condition:
    # do something
    if break_condition:
        break           # Exit loop
    if skip_condition:
        continue        # Skip to next iteration
```

## Functions

### Definition
```python
def function_name(param1, param2=default):
    """Docstring describing function"""
    # function body
    return value

# Lambda (anonymous) function
square = lambda x: x**2
```

### Special Parameters
```python
def func(*args):        # Variable positional args
    print(args)         # Tuple of arguments

def func(**kwargs):     # Variable keyword args
    print(kwargs)       # Dict of arguments

func(1, 2, x=3, y=4)
```

## Classes

### Basic Class
```python
class MyClass:
    def __init__(self, value):
        self.value = value

    def method(self):
        return self.value

obj = MyClass(10)
obj.method()
```

### Inheritance
```python
class Child(Parent):
    def __init__(self, value):
        super().__init__(value)

    def method(self):
        return super().method() * 2
```

## File I/O

### Reading Files
```python
# Read entire file
with open("file.txt", "r") as f:
    content = f.read()

# Read line by line
with open("file.txt", "r") as f:
    for line in f:
        print(line.strip())
```

### Writing Files
```python
with open("file.txt", "w") as f:
    f.write("Hello\n")
    f.writelines(["Line1\n", "Line2\n"])
```

### JSON
```python
import json

# Write JSON
with open("data.json", "w") as f:
    json.dump(data, f, indent=2)

# Read JSON
with open("data.json", "r") as f:
    data = json.load(f)
```

## Exception Handling

```python
try:
    risky_operation()
except SpecificError as e:
    handle_error(e)
except Exception as e:
    handle_general_error(e)
else:
    # Runs if no exception
    success_action()
finally:
    # Always runs
    cleanup()

# Raise exceptions
raise ValueError("Invalid value")
```

## Common Modules

### Imports
```python
import module
from module import function
from module import function as fn
import module as mod
```

### datetime
```python
from datetime import datetime, timedelta

now = datetime.now()
date = datetime(2024, 1, 15)
formatted = now.strftime("%Y-%m-%d")
parsed = datetime.strptime("2024-01-15", "%Y-%m-%d")
tomorrow = now + timedelta(days=1)
```

### random
```python
import random

random.random()         # Float [0.0, 1.0)
random.randint(1, 10)   # Int [1, 10]
random.choice([1,2,3])  # Pick random item
random.shuffle(list)    # Shuffle in place
```

### math
```python
import math

math.sqrt(16)           # 4.0
math.ceil(3.2)          # 4
math.floor(3.8)         # 3
math.pi                 # 3.14159...
math.pow(2, 3)          # 8.0
```

### os
```python
import os

os.getcwd()             # Current directory
os.listdir()            # List files
os.path.exists(path)    # Check if exists
os.path.join(a, b)      # Join paths
os.mkdir(path)          # Create directory
```

## Built-in Functions

```python
len(obj)                # Length
max(list)               # Maximum
min(list)               # Minimum
sum(list)               # Sum
abs(-5)                 # Absolute value
round(3.14159, 2)       # Round: 3.14
range(start, stop, step) # Range of numbers
enumerate(list)         # (index, value) pairs
zip(list1, list2)       # Combine lists
map(func, list)         # Apply function
filter(func, list)      # Filter elements
all([True, True])       # All true: True
any([False, True])      # Any true: True
sorted(list)            # Return sorted
reversed(list)          # Return reversed
```

## List/Dict/Set Comprehensions

```python
# List comprehension
[expr for item in iterable if condition]

# Dict comprehension
{key: value for item in iterable if condition}

# Set comprehension
{expr for item in iterable if condition}

# Generator expression
(expr for item in iterable if condition)
```

## Common Patterns

### Swap Variables
```python
a, b = b, a
```

### Check Multiple Conditions
```python
if x in [1, 2, 3, 4, 5]:
    # do something
```

### Default Dictionary Value
```python
value = dict.get(key, default_value)
```

### Counter Pattern
```python
count = {}
for item in items:
    count[item] = count.get(item, 0) + 1
```

### Filter List
```python
filtered = [x for x in list if condition]
```

### Find Max/Min with Key
```python
max(list, key=lambda x: x.attribute)
```

## String Escape Characters

```python
\n      # Newline
\t      # Tab
\\      # Backslash
\'      # Single quote
\"      # Double quote
\r      # Carriage return
```

## Useful Commands

### Running Python
```bash
python3 script.py       # Run script
python3                 # Interactive mode
python3 -m module       # Run module
```

### pip (Package Manager)
```bash
pip3 install package    # Install package
pip3 list               # List installed
pip3 freeze             # Export requirements
pip3 install -r requirements.txt  # Install from file
```

## Tips & Best Practices

1. **PEP 8**: Follow Python style guide
2. **Naming**: `snake_case` for variables/functions, `PascalCase` for classes
3. **Indentation**: Use 4 spaces (not tabs)
4. **Docstrings**: Document functions and classes
5. **DRY**: Don't Repeat Yourself - use functions
6. **Error Handling**: Use try-except for risky operations
7. **List Comprehensions**: More readable than loops for simple operations
8. **f-strings**: Modern string formatting
9. **Context Managers**: Use `with` for files
10. **Virtual Environments**: Isolate project dependencies

---

*Keep this reference handy while coding. Print it out or keep it open in another window!*
