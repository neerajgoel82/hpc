# Phase 4: Advanced Python (Weeks 9-11)

Time to level up with advanced concepts that make Python truly powerful!

## What You'll Learn

- Iterators and generators
- Decorators
- Context managers
- Advanced list comprehensions
- Regular expressions
- Working with dates/times
- Testing and debugging

## Week-by-Week Breakdown

### Week 9: Iterators, Generators & Decorators
- `42_iterators.py` - Iterator protocol and __iter__
- `43_generators.py` - yield keyword and generators
- `44_decorators_basics.py` - Function decorators
- `45_decorators_advanced.py` - Decorators with parameters
- `46_args_kwargs.py` - *args and **kwargs
- **Project:** `project_data_pipeline.py` - Data processing pipeline

### Week 10: Working with Data
- `47_regex.py` - Regular expressions
- `48_datetime.py` - Date and time handling
- `49_collections.py` - defaultdict, Counter, namedtuple
- `50_comprehensions_advanced.py` - Advanced comprehensions
- `51_file_operations.py` - Advanced file operations
- **Project:** `project_log_analyzer.py` - Log file analyzer

### Week 11: Testing & Best Practices
- `52_debugging.py` - Debugging techniques
- `53_unittest.py` - Unit testing with unittest
- `54_pytest.py` - Testing with pytest
- `55_logging.py` - Logging module
- `56_best_practices.py` - Code quality and style
- **Project:** `project_tested_library.py` - Create tested library

## Key Concepts

### Generators
```python
def countdown(n):
    while n > 0:
        yield n
        n -= 1

for num in countdown(5):
    print(num)
```

### Decorators
```python
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Took {end - start} seconds")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)
```

### Regular Expressions
```python
import re

pattern = r'\d{3}-\d{3}-\d{4}'
text = "Call me at 555-123-4567"
match = re.search(pattern, text)
if match:
    print(f"Found: {match.group()}")
```

## Why Learn Advanced Python?

- **Efficiency**: Generators save memory
- **Clean code**: Decorators add functionality elegantly
- **Reliability**: Testing ensures code works
- **Professional**: These are industry-standard practices

## Getting Started

```bash
cd phase4-advanced
python3 42_iterators.py
```

## Prerequisites

- Completed Phases 1-3
- Strong OOP understanding
- Comfortable with functions and modules

Ready for advanced Python? Let's dive in!
