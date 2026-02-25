# Python Learning Journey

A comprehensive, hands-on curriculum for learning Python from beginner to advanced level, designed for HPC (High Performance Computing) applications.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Directory Structure](#directory-structure)
- [Phase Structure](#phase-structure)
- [Learning Paths](#learning-paths)
- [Running Python Programs](#running-python-programs)
- [Phase Completion Checklist](#phase-completion-checklist)
- [Timeline Estimates](#timeline-estimates)
- [Tools and Resources](#tools-and-resources)
- [Tips for Success](#tips-for-success)
- [Quick Reference](#quick-reference)

## Overview

This curriculum takes you from Python basics to advanced concepts through a structured, project-based approach. Each phase builds on previous knowledge with hands-on exercises and real projects.

**Target Audience**: Beginners to intermediate programmers
**Prerequisites**: None - start from scratch
**Time Commitment**: 30 minutes to 2 hours daily
**Total Duration**: 11-12 weeks (full-time) to 6-12 months (part-time)

## Quick Start

### 1. Verify Python Installation

```bash
# Check Python version (requires 3.8+)
python3 --version

# Check pip is available
pip3 --version
```

### 2. Install Python (if needed)

**macOS:**
```bash
# Use Homebrew
brew install python3
```

**Windows:**
- Download from [python.org](https://www.python.org/downloads/)
- Check "Add Python to PATH" during installation

**Linux:**
```bash
sudo apt-get update
sudo apt-get install python3 python3-pip
```

### 3. Run Your First Program

```bash
# Navigate to samples directory
cd phase1-foundations

# Run hello world
python3 01_hello_world.py

# You should see: Hello, World!
```

### 4. Choose Your Editor

**Recommended:**
- **VS Code** (Most popular) - [code.visualstudio.com](https://code.visualstudio.com/)
  - Install Python extension
  - Install Pylance extension
- **PyCharm Community** (Full IDE) - [jetbrains.com](https://www.jetbrains.com/pycharm/)

**Simple Alternatives:**
- IDLE (comes with Python)
- Sublime Text
- Atom

## Directory Structure

```
python/samples/
├── phase1-foundations/       # Weeks 1-3: Basics
│   ├── 01_hello_world.py
│   ├── 02_variables.py
│   ├── ...
│   └── projects/
├── phase2-intermediate/      # Weeks 4-6: Functions, Files, Modules
│   ├── 01_functions.py
│   ├── ...
│   └── projects/
├── phase3-oop/              # Weeks 7-8: Object-Oriented Programming
│   ├── 01_classes.py
│   ├── ...
│   └── projects/
├── phase4-advanced/         # Weeks 9-11: Advanced concepts
│   ├── 01_iterators.py
│   ├── ...
│   └── projects/
├── phase5-specialization/   # Week 12+: Choose your path
│   ├── data-science/
│   ├── web-dev/
│   ├── automation/
│   └── cli-tools/
└── README.md               # This file
```

## Phase Structure

### Phase 1: Foundations (Weeks 1-3)
**Goal**: Master Python basics and fundamental programming concepts

**Topics Covered:**
- Variables and data types (int, float, str, bool)
- Operators (arithmetic, comparison, logical)
- Input/Output operations
- String manipulation and formatting (f-strings)
- Control flow (if/elif/else statements)
- Loops (while, for, range)
- Loop control (break, continue)
- Data structures (lists, tuples, dictionaries, sets)
- List/dict comprehensions

**Weekly Breakdown:**

**Week 1: Getting Started**
- Hello World
- Variables & Types
- Operators
- Input/Output
- Strings
- **Project**: Calculator

**Week 2: Control Flow**
- Conditionals (if/elif/else)
- While Loops
- For Loops
- Loop Control
- String Formatting
- **Project**: Guessing Game

**Week 3: Data Structures**
- Lists
- Tuples
- Dictionaries
- Sets
- Comprehensions
- **Project**: Contact Book

**Checkpoint**: Can you build a simple program with user input, loops, and data storage?

---

### Phase 2: Intermediate (Weeks 4-6)
**Goal**: Write reusable, robust code with functions and file handling

**Topics Covered:**
- Function definition and parameters
- Return values and scope
- Lambda functions
- Recursion
- Exception handling (try/except/finally)
- File I/O operations (read/write)
- Context managers (with statement)
- CSV and JSON file handling
- Modules and imports
- Standard library exploration
- Creating custom modules
- Packages and package structure
- Virtual environments

**Weekly Breakdown:**

**Week 4: Functions**
- Function Basics
- Parameters & Arguments
- Return Values
- Scope (local, global, nonlocal)
- Lambda Functions
- Recursion
- **Project**: Text Analyzer

**Week 5: Error Handling & Files**
- Exceptions and error types
- Try/except/finally blocks
- File Reading
- File Writing
- CSV Files
- JSON Files
- **Project**: File Organizer

**Week 6: Modules & Packages**
- Importing modules
- Standard Library
- Creating Modules
- Packages
- Virtual Environments
- **Project**: CLI Tool

**Checkpoint**: Can you create a program with functions, error handling, and file I/O?

---

### Phase 3: Object-Oriented Programming (Weeks 7-8)
**Goal**: Think in objects and design class hierarchies

**Topics Covered:**
- Classes and objects
- Attributes and methods
- Constructors (`__init__`)
- Instance vs class variables
- Encapsulation (public, private)
- Inheritance and super()
- Polymorphism
- Magic/dunder methods (`__str__`, `__repr__`, etc.)
- Properties and decorators
- Composition vs inheritance
- Abstract base classes

**Weekly Breakdown:**

**Week 7: OOP Basics**
- Classes & Objects
- Attributes & Methods
- Constructors
- Class Variables
- Encapsulation
- **Project**: Bank Account System

**Week 8: Advanced OOP**
- Inheritance
- Polymorphism
- Magic Methods
- Properties
- Composition
- **Project**: Game Characters

**Checkpoint**: Can you design and implement a class hierarchy?

---

### Phase 4: Advanced Python (Weeks 9-11)
**Goal**: Master advanced Python concepts and professional practices

**Topics Covered:**
- Iterators and iteration protocol
- Generators and yield
- Decorator basics and advanced patterns
- *args and **kwargs
- Regular expressions (regex)
- Date and time handling
- Collections module (Counter, defaultdict, etc.)
- Advanced comprehensions
- Debugging techniques
- Unit testing (unittest/pytest)
- Logging
- Best practices and PEP 8
- Type hints

**Weekly Breakdown:**

**Week 9: Iterators, Generators & Decorators**
- Iterators
- Generators
- Decorators Basics
- Advanced Decorators
- *args & **kwargs
- **Project**: Data Pipeline

**Week 10: Working with Data**
- Regular Expressions
- Date/Time
- Collections Module
- Advanced Comprehensions
- File Operations
- **Project**: Log Analyzer

**Week 11: Testing & Best Practices**
- Debugging
- Unit Testing
- Pytest
- Logging
- Best Practices
- **Project**: Tested Library

**Checkpoint**: Can you write tested, professional Python code?

---

### Phase 5: Specialization (Week 12+)
**Goal**: Specialize in your area of interest

**Choose Your Path(s):**

#### Path A: Web Development
- Flask/Django frameworks
- REST APIs
- Databases (SQLite, PostgreSQL)
- Authentication
- Deployment

#### Path B: Data Science (HPC Focus)
- NumPy & Pandas
- Data cleaning and transformation
- Matplotlib/Seaborn
- Jupyter notebooks
- Basic statistics
- Performance optimization

#### Path C: Automation
- Web scraping (BeautifulSoup, Selenium)
- API interaction
- Task scheduling
- Email automation
- System scripts

#### Path D: CLI Tools
- argparse/Click
- Rich output formatting
- Configuration management
- Package distribution
- Tool building

#### Path E: API Development
- FastAPI
- Authentication
- Databases
- API design
- Documentation (Swagger)

**Checkpoint**: Complete a portfolio project in your chosen path!

## Learning Paths

### Approach 1: Standard Path (Recommended)
Follow phases 1-5 in order, completing all exercises and projects.

**Time**: 3 months full-time, 6-12 months part-time
**Best for**: Complete beginners, structured learners

### Approach 2: Fast Track
Complete only core exercises in each phase, skip optional challenges.

**Time**: 6-8 weeks full-time, 3-6 months part-time
**Best for**: Programmers with experience in other languages

### Approach 3: Project-Focused
Complete exercises as needed to support project work.

**Time**: Variable
**Best for**: Learn-by-doing types, those with specific goals

### Approach 4: HPC-Focused
Standard path + emphasis on NumPy, performance, and Phase 5 Data Science.

**Time**: 4 months full-time
**Best for**: Those interested in scientific computing

## Running Python Programs

### Method 1: Direct Execution
```bash
# Run a Python script
python3 filename.py

# Run with arguments
python3 script.py arg1 arg2
```

### Method 2: Interactive Mode (REPL)
```bash
# Start Python interactive shell
python3

# Try some code
>>> print("Hello")
Hello
>>> x = 10
>>> x * 2
20
>>> exit()
```

### Method 3: In Your Editor
- **VS Code**: Click Run button or press F5
- **PyCharm**: Right-click file → Run
- **IDLE**: Run → Run Module (F5)

### Method 4: Jupyter Notebooks
```bash
# Install Jupyter
pip3 install jupyter

# Start notebook server
jupyter notebook

# Opens in browser
```

### Running with Virtual Environment
```bash
# Create virtual environment
python3 -m venv .venv

# Activate
source .venv/bin/activate  # Unix/macOS
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run your script
python script.py

# Deactivate when done
deactivate
```

## Phase Completion Checklist

### Core Python Skills
- [ ] Variables and data types
- [ ] Operators and expressions
- [ ] Control flow (if/else, loops)
- [ ] Functions and parameters
- [ ] Data structures (list, dict, set, tuple)
- [ ] String manipulation
- [ ] File I/O
- [ ] Error handling
- [ ] Classes and OOP
- [ ] Modules and packages

### Advanced Skills
- [ ] Generators and iterators
- [ ] Decorators
- [ ] Context managers
- [ ] Regular expressions
- [ ] Testing (unittest/pytest)
- [ ] Virtual environments
- [ ] Package management (pip)
- [ ] Debugging techniques

### Professional Skills
- [ ] Code organization
- [ ] Documentation (docstrings)
- [ ] Version control (git)
- [ ] Code style (PEP 8)
- [ ] Problem-solving
- [ ] Reading documentation
- [ ] Debugging skills
- [ ] Project planning

### Phase Progress Tracking
- [ ] Phase 1: Foundations (3 weeks)
- [ ] Phase 2: Intermediate (3 weeks)
- [ ] Phase 3: OOP (2 weeks)
- [ ] Phase 4: Advanced (3 weeks)
- [ ] Phase 5: Specialization (ongoing)

### Learning Milestones
- [ ] Milestone 1: First Program - Complete "Hello World"
- [ ] Milestone 2: First Project - Calculator or Guessing Game
- [ ] Milestone 3: Data Structures Master - Complete Phase 1
- [ ] Milestone 4: Function Expert - Complete Phase 2
- [ ] Milestone 5: OOP Practitioner - Complete Phase 3
- [ ] Milestone 6: Advanced Developer - Complete Phase 4
- [ ] Milestone 7: Specialist - Complete specialization path

## Timeline Estimates

### Full-Time Learning (40 hrs/week)
- **Phase 1**: 2 weeks (60-90 hours)
- **Phase 2**: 2 weeks (60-90 hours)
- **Phase 3**: 2 weeks (40-60 hours)
- **Phase 4**: 3 weeks (60-90 hours)
- **Phase 5**: Ongoing
- **Total**: 11-12 weeks to job-ready

### Part-Time Learning (10 hrs/week)
- **Phase 1**: 6 weeks
- **Phase 2**: 6 weeks
- **Phase 3**: 6 weeks
- **Phase 4**: 10 weeks
- **Phase 5**: Ongoing
- **Total**: 6-12 months to job-ready

### Casual Learning (5 hrs/week)
- Double the part-time timeline
- Focus on understanding over speed
- Take breaks when needed

### Skill Development Timeline

**Month 1: Beginner**
- Comfortable with Python syntax
- Can write simple programs
- Understand basic concepts

**Month 2: Advanced Beginner**
- Writing functions and classes
- Handling errors properly
- Working with files

**Month 3: Intermediate**
- Using OOP effectively
- Understanding advanced concepts
- Building complete programs

**Month 4+: Advanced**
- Specializing in chosen path
- Contributing to open source
- Building portfolio projects

## Tools and Resources

### Development Environment

**IDEs and Editors:**
- VS Code (recommended) - Lightweight, extensible
- PyCharm Community - Full-featured Python IDE
- Jupyter Notebook - Interactive notebooks
- IDLE - Built-in Python editor
- Sublime Text - Fast text editor
- Atom - Hackable editor

**Essential VS Code Extensions:**
- Python (Microsoft)
- Pylance (Microsoft)
- Python Indent
- autoDocstring
- GitLens

### Python Libraries by Phase

**Phase 1-2 (Built-in):**
- No external libraries needed
- Standard library only

**Phase 3-4:**
- pytest - Testing framework
- black - Code formatter
- flake8 - Linter

**Phase 5 (Data Science):**
- numpy - Numerical computing
- pandas - Data analysis
- matplotlib - Plotting
- jupyter - Notebooks

**Phase 5 (Web):**
- flask/django - Web frameworks
- requests - HTTP client
- fastapi - Modern API framework

**Phase 5 (Automation):**
- beautifulsoup4 - Web scraping
- selenium - Browser automation
- schedule - Task scheduling

### Documentation Resources

**Official:**
- [Python Docs](https://docs.python.org/3/) - Official documentation
- [Python Tutorial](https://docs.python.org/3/tutorial/) - Official tutorial
- [PEP 8](https://pep8.org/) - Style guide

**Learning Platforms:**
- [Real Python](https://realpython.com/) - Tutorials and articles
- [Python.org Beginner's Guide](https://wiki.python.org/moin/BeginnersGuide)
- [W3Schools Python](https://www.w3schools.com/python/)

### Practice Platforms

**Coding Challenges:**
- [LeetCode](https://leetcode.com/) - Algorithm practice
- [HackerRank](https://hackerrank.com/) - Challenges
- [Codewars](https://codewars.com/) - Kata exercises
- [Exercism](https://exercism.io/tracks/python) - Mentored learning

**Interactive Learning:**
- [Python Tutor](http://pythontutor.com/) - Visualize code execution
- [Codecademy Python](https://www.codecademy.com/learn/learn-python-3)
- [DataCamp](https://www.datacamp.com/) - Data science focus

### Community Resources

**Forums and Q&A:**
- Reddit: [r/learnpython](https://reddit.com/r/learnpython)
- Stack Overflow: Tag [python]
- Python Discord server
- Python Forum: [python-forum.io](https://python-forum.io/)

**YouTube Channels:**
- Corey Schafer - Clear tutorials
- Tech With Tim - Projects and guides
- Real Python - Professional content
- Programming with Mosh - Beginner-friendly

**Books:**
- "Python Crash Course" by Eric Matthes
- "Automate the Boring Stuff" by Al Sweigart
- "Fluent Python" by Luciano Ramalho
- "Effective Python" by Brett Slatkin

## Tips for Success

### How to Use This Curriculum

1. **Start with Phase 1** - Don't skip ahead even if you know some basics
2. **Complete exercises in order** - Each builds on previous knowledge
3. **Type all code yourself** - No copy-pasting!
4. **Run and test everything** - Make sure you understand the output
5. **Experiment** - Modify the code, break it, fix it
6. **Check solutions only after trying** - Solutions are in `_solutions/` folders

### Daily Learning Routine

**Minimum (30 minutes/day):**
1. Read through one exercise file (10 min)
2. Complete the TODOs and exercises (15 min)
3. Experiment and modify code (5 min)

**Recommended (1-2 hours/day):**
1. Read and understand the lesson (20 min)
2. Complete all exercises (40 min)
3. Try bonus challenges (20 min)
4. Build something using concepts (20 min)

### Weekly Structure

**Week Pattern:**
- Days 1-5: Complete daily exercises
- Day 6: Review week's concepts
- Day 7: Complete week's project

### Learning Best Practices

#### 1. Type, Don't Copy
Always type code yourself. Muscle memory helps!

#### 2. Run Every Example
Execute code after every few lines. See what happens!

#### 3. Break Things
Intentionally make errors. Learn from them!

#### 4. Experiment
Modify examples. Try different inputs. Explore!

#### 5. Build Projects
Apply concepts immediately. Build real things!

### When You're Stuck

**Follow This Order:**
1. **Read the error message** - It tells you what's wrong!
2. **Check your syntax** - Missing colons, parentheses, indentation?
3. **Print debug info** - Add print() statements
4. **Review the lesson** - Re-read relevant section
5. **Search online** - Google the error message
6. **Ask for help** - Stack Overflow, Reddit r/learnpython

### Common Beginner Mistakes

#### Indentation Errors
```python
# Wrong
def greet():
print("Hello")  # IndentationError!

# Correct
def greet():
    print("Hello")  # Indented with 4 spaces
```

#### Missing Colons
```python
# Wrong
if x > 5
    print(x)

# Correct
if x > 5:
    print(x)
```

#### Wrong Quote Types
```python
# Wrong
print('It's a nice day')  # SyntaxError

# Correct
print("It's a nice day")
print('It\'s a nice day')  # Escape quote
```

#### Comparing vs Assigning
```python
# Wrong
if x = 5:  # SyntaxError: trying to assign

# Correct
if x == 5:  # Comparing with ==
```

### Staying Motivated

#### Set Goals
- "Complete Phase 1 by end of month"
- "Build 3 projects this quarter"
- "Contribute to open source by June"

#### Celebrate Wins
- Completed first program
- Fixed first bug
- Built first project
- Helped someone else

#### Join a Community
- Find study buddies
- Share your progress
- Help others learn

#### Mix It Up
- Watch tutorials
- Read articles
- Code along
- Build projects
- Teach others

### Progress Tracking

**Keep a Learning Journal:**
Create `my_progress.md`:
```markdown
# My Python Learning Journal

## Week 1
- Completed: Exercises 1-5
- Struggles: Understanding loops
- Wins: Built calculator!
- Next: Start Week 2

## Week 2
...
```

**Build a Portfolio:**
Create a `my_projects/` folder:
```
my_projects/
├── calculator/
├── guessing_game/
├── contact_book/
└── ...
```

## Quick Reference

### Basic Syntax

```python
# Print & Comments
print("Hello")              # Print to console
# This is a comment
"""This is a
multi-line comment"""

# Variables
name = "Alice"              # String
age = 25                    # Integer
price = 19.99               # Float
is_active = True            # Boolean
nothing = None              # None type
```

### Operators

```python
# Arithmetic
10 + 5      # Addition: 15
10 - 5      # Subtraction: 5
10 * 5      # Multiplication: 50
10 / 5      # Division: 2.0
10 // 3     # Floor division: 3
10 % 3      # Modulus: 1
10 ** 2     # Exponentiation: 100

# Comparison
10 == 10    # Equal: True
10 != 5     # Not equal: True
10 > 5      # Greater than: True
10 < 5      # Less than: False
10 >= 10    # Greater or equal: True
10 <= 10    # Less or equal: True

# Logical
True and False  # False
True or False   # True
not True        # False
```

### Strings

```python
s = "Hello"
s + " World"            # Concatenation: "Hello World"
s * 3                   # Repetition: "HelloHelloHello"
s[0]                    # Index: "H"
s[-1]                   # Last char: "o"
s[1:4]                  # Slice: "ell"
len(s)                  # Length: 5

# Methods
s.upper()               # "HELLO"
s.lower()               # "hello"
s.strip()               # Remove whitespace
s.replace("l", "L")     # Replace chars
s.split()               # Split into list

# Formatting
name = "Alice"
age = 25
f"I'm {name}, {age}"    # f-string: "I'm Alice, 25"
```

### Lists

```python
lst = [1, 2, 3, 4, 5]
lst[0]                  # First: 1
lst[-1]                 # Last: 5
lst[1:3]                # Slice: [2, 3]

# Methods
lst.append(6)           # Add to end
lst.insert(0, 0)        # Insert at position
lst.remove(3)           # Remove first 3
lst.pop()               # Remove & return last
lst.sort()              # Sort in place

# Comprehension
[x**2 for x in range(5)]              # [0,1,4,9,16]
[x for x in range(10) if x % 2 == 0]  # [0,2,4,6,8]
```

### Dictionaries

```python
d = {"name": "Alice", "age": 25}
d["name"]               # Access value: "Alice"
d.get("city", "NYC")    # Get with default
d["city"] = "Paris"     # Add/update

# Methods
d.keys()                # All keys
d.values()              # All values
d.items()               # Key-value pairs
"name" in d             # Check key exists

# Comprehension
{x: x**2 for x in range(5)}  # {0:0, 1:1, 2:4, 3:9, 4:16}
```

### Control Flow

```python
# If statements
if condition:
    # do something
elif other_condition:
    # do something else
else:
    # default action

# For loops
for i in range(5):      # 0,1,2,3,4
    print(i)

for item in list:       # Iterate list
    print(item)

for i, item in enumerate(list):  # With index
    print(i, item)

# While loops
while condition:
    # do something
    if break_condition:
        break           # Exit loop
    if skip_condition:
        continue        # Skip to next
```

### Functions

```python
# Definition
def function_name(param1, param2=default):
    """Docstring describing function"""
    return value

# Lambda
square = lambda x: x**2

# Variable arguments
def func(*args, **kwargs):
    print(args)         # Tuple
    print(kwargs)       # Dict
```

### Classes

```python
# Basic class
class MyClass:
    def __init__(self, value):
        self.value = value

    def method(self):
        return self.value

obj = MyClass(10)

# Inheritance
class Child(Parent):
    def __init__(self, value):
        super().__init__(value)
```

### File I/O

```python
# Reading
with open("file.txt", "r") as f:
    content = f.read()

# Writing
with open("file.txt", "w") as f:
    f.write("Hello\n")

# JSON
import json
with open("data.json", "w") as f:
    json.dump(data, f, indent=2)
```

### Exception Handling

```python
try:
    risky_operation()
except SpecificError as e:
    handle_error(e)
except Exception as e:
    handle_general_error(e)
else:
    success_action()
finally:
    cleanup()
```

### Built-in Functions

```python
len(obj)                # Length
max(list)               # Maximum
min(list)               # Minimum
sum(list)               # Sum
abs(-5)                 # Absolute value
round(3.14159, 2)       # Round: 3.14
range(start, stop, step) # Range
enumerate(list)         # (index, value) pairs
zip(list1, list2)       # Combine lists
sorted(list)            # Return sorted
```

### Common Patterns

```python
# Swap variables
a, b = b, a

# Check multiple conditions
if x in [1, 2, 3, 4, 5]:
    pass

# Default dictionary value
value = dict.get(key, default_value)

# Filter list
filtered = [x for x in list if condition]
```

### pip Commands

```bash
pip3 install package    # Install package
pip3 list               # List installed
pip3 freeze             # Export requirements
pip3 install -r requirements.txt  # Install from file
```

## What's Next After This Curriculum?

### Continue Learning
1. **Contribute to Open Source** - Find projects on GitHub
2. **Build Portfolio Projects** - Create 3-5 substantial projects
3. **Learn Advanced Topics** - Async, networking, performance
4. **Explore Frameworks** - Django, Flask, FastAPI, etc.
5. **Study Algorithms** - LeetCode, HackerRank practice

### Career Paths
1. **Junior Developer** - Entry-level positions
2. **Python Developer** - Specialized roles
3. **Full-Stack Developer** - Front + back end
4. **Data Analyst/Scientist** - Data-focused roles
5. **DevOps Engineer** - Automation & infrastructure
6. **Software Engineer** - General development

### Keep Growing
- Read Python books
- Follow Python blogs/podcasts
- Attend conferences/meetups
- Teach others (best way to learn!)
- Never stop building!

---

## Additional Notes

**Remember**: Consistency beats intensity. 30 minutes daily is better than 3 hours once a week!

**Code Quality**: This is an HPC repository - consider performance, memory usage, and optimization opportunities as you progress.

**Cross-Language Learning**: Compare Python implementations with C/C++/CUDA versions in this repository to understand performance trade-offs.

Ready to start? Head to `phase1-foundations/` now!
