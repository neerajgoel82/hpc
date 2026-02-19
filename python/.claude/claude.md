# Python Programming - Claude Instructions

## Curriculum Structure

Python samples are organized by phases:
- **phase1-foundations**: Basic syntax, data types, control flow, functions
- **phase2-intermediate**: Data structures, file I/O, modules, exceptions
- **phase3-oop**: Classes, inheritance, polymorphism
- **phase4-advanced**: Decorators, generators, context managers, advanced topics
- **phase5-specialization**: Domain-specific topics (data science, async, etc.)

## Python Standards

### Version
- Python 3.9+ required
- Use modern Python features (f-strings, type hints, etc.)

### Style Guide
- Follow PEP 8
- Use `black` formatter if available
- Maximum line length: 88 characters (black default) or 79 (PEP 8)

## Coding Style

### Naming Conventions
- Functions/variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_CASE`
- Private attributes: `_leading_underscore`

### Type Hints (Phase 3+)
```python
def calculate_sum(numbers: list[int]) -> int:
    return sum(numbers)

def process_data(data: dict[str, Any]) -> None:
    pass
```

### Imports
```python
# Order:
1. Standard library imports
2. Related third party imports
3. Local application imports

# Within each group, alphabetically
import os
import sys

import numpy as np
import pandas as pd

from .local_module import something
```

## Phase-Specific Guidelines

### Phase 1: Foundations
- Keep simple and clear
- No type hints required
- Focus on Python basics
- Can ignore advanced error handling
- Simple print() for output

### Phase 2: Intermediate
- Introduce proper error handling with try/except
- Use context managers for file operations
- Demonstrate Pythonic patterns

### Phase 3: OOP
- Use type hints for class methods
- Demonstrate proper class design
- Use dataclasses when appropriate
- Show `__str__`, `__repr__` implementations

### Phase 4-5: Advanced
- Type hints required
- Use advanced Python features
- Consider performance (use appropriate data structures)
- Add docstrings to functions and classes

## Common Patterns

### Error Handling
```python
# Phase 1-2: Basic
try:
    result = operation()
except Exception as e:
    print(f"Error: {e}")

# Phase 3+: Specific exceptions
try:
    with open(filename) as f:
        data = f.read()
except FileNotFoundError:
    print(f"File {filename} not found")
except IOError as e:
    print(f"IO error: {e}")
```

### File Operations
```python
# Always use context managers (Phase 2+)
with open("file.txt", "r") as f:
    content = f.read()

# Not:
f = open("file.txt")
content = f.read()
f.close()
```

### String Formatting
```python
# Use f-strings (Python 3.6+)
name = "Alice"
age = 30
print(f"{name} is {age} years old")

# Not .format() or % unless teaching compatibility
```

### List Comprehensions
```python
# Prefer when readable
squares = [x**2 for x in range(10)]

# Use regular loops when complex
results = []
for item in data:
    if complex_condition(item):
        processed = complex_processing(item)
        results.append(processed)
```

## HPC-Specific Python

### Performance Libraries
- **NumPy**: For numerical computing
- **Numba**: For JIT compilation
- **Cython**: For C extensions
- **multiprocessing**: For parallel processing

### NumPy Best Practices
```python
import numpy as np

# Vectorized operations (fast)
arr = np.array([1, 2, 3, 4, 5])
result = arr * 2

# Not loops (slow)
result = [x * 2 for x in arr]

# Specify dtypes for performance
arr = np.zeros(1000000, dtype=np.float32)
```

### Profiling
```python
# Use timeit for benchmarking
import timeit
time = timeit.timeit(lambda: function(), number=1000)

# Use cProfile for detailed profiling
import cProfile
cProfile.run('function()')
```

## What NOT to Do

- Don't use `import *` (except in notebooks for teaching)
- Don't use mutable default arguments
- Don't compare booleans with `== True`
- Don't use `eval()` or `exec()` (security risk)
- Don't ignore exceptions silently (`except: pass`)
- Don't use string concatenation in loops (use join)
- Don't check types with `type()`, use `isinstance()`

## When Adding New Samples
1. Place in appropriate phase directory
2. Match complexity level of the phase
3. Follow PEP 8 style guide
4. Test that it runs without errors
5. Add type hints for phase 3+
6. Update phase README if needed

## Testing
```bash
# Run the script
python3 script.py

# Check syntax/style (if tools available)
python3 -m py_compile script.py
flake8 script.py
black --check script.py
```

## Virtual Environment
```bash
# Create virtual environment
python3 -m venv .venv

# Activate
source .venv/bin/activate  # Unix
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Dependencies
- Keep `requirements.txt` updated
- Pin versions for reproducibility when needed
- Separate requirements for different purposes:
  - `requirements.txt`: Core dependencies
  - `requirements-dev.txt`: Development tools (optional)

## Notebooks
- Place in `notebooks/` directory within python/
- Use clear cell outputs
- Add markdown cells for explanations
- Keep notebooks focused on one topic
- Include imports in first cell

## Documentation

### Docstrings (Phase 3+)
```python
def function(param1: int, param2: str) -> bool:
    """
    Brief description of function.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value
    """
    pass
```

### Comments
- Use for complex logic
- Explain "why" not "what"
- Keep comments updated with code

## Performance Tips for HPC

1. **Use NumPy for numerical operations**
2. **Avoid Python loops for large datasets** (vectorize)
3. **Profile before optimizing** (`cProfile`, `line_profiler`)
4. **Use appropriate data structures** (set for membership, dict for lookups)
5. **Consider Numba for numerical code**
6. **Use multiprocessing for CPU-bound tasks**
7. **Understand the GIL** (Global Interpreter Lock) limitations

## Comparing with C/C++/CUDA
When creating Python versions of C/C++/CUDA code:
- Show both pure Python and NumPy versions
- Document performance differences
- Explain when Python is appropriate vs C/CUDA
- Consider hybrid approaches (Python + C extensions)
