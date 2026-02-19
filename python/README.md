# Python Programming

Learn Python programming from fundamentals through advanced topics, with focus on HPC and scientific computing.

## Structure

```
python/
├── samples/          # Learning samples organized by phases
├── projects/         # Complete Python applications
├── notebooks/        # Python-related notebooks (if needed)
└── requirements.txt  # Python dependencies
```

## Curriculum

The samples are organized into progressive learning phases:

### Phase 1: Foundations
- Basic syntax and structure
- Data types (strings, numbers, lists, etc.)
- Control flow (if, loops)
- Functions and scope

### Phase 2: Intermediate
- Data structures (dict, set, tuple)
- File I/O and exceptions
- Modules and packages
- List comprehensions

### Phase 3: Object-Oriented Programming
- Classes and objects
- Inheritance and polymorphism
- Magic methods
- Decorators basics

### Phase 4: Advanced
- Decorators and generators
- Context managers
- Iterators and iterables
- Advanced Python features

### Phase 5: Specialization
- Domain-specific topics
- Scientific computing (NumPy, SciPy)
- Async programming
- Performance optimization

## Getting Started

### Prerequisites
```bash
# Check Python version (should be 3.9+)
python3 --version

# Install pip if needed
python3 -m ensurepip --upgrade
```

### Setup Virtual Environment
```bash
# Create virtual environment
python3 -m venv .venv

# Activate
source .venv/bin/activate  # Unix/macOS
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Deactivate when done
deactivate
```

### Running Samples
```bash
# Run a single script
python3 script.py

# Or within virtual environment
python script.py
```

## Code Style

Follow PEP 8 style guide:

### Naming Conventions
- **Variables/Functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_CASE`
- **Private**: `_leading_underscore`

### Type Hints (Phase 3+)
```python
def calculate_sum(numbers: list[int]) -> int:
    return sum(numbers)

def greet(name: str) -> None:
    print(f"Hello, {name}!")
```

### Modern Python Features
- Use f-strings for formatting
- List comprehensions when readable
- Context managers for resources
- Type hints for clarity

## HPC with Python

### Key Libraries

#### NumPy - Numerical Computing
```python
import numpy as np

# Vectorized operations (fast)
arr = np.array([1, 2, 3, 4, 5])
result = arr * 2  # Much faster than loops

# Specify data types
arr = np.zeros(1000000, dtype=np.float32)
```

#### Numba - JIT Compilation
```python
from numba import jit

@jit(nopython=True)
def fast_function(x):
    # This will be compiled to machine code
    return x * x + 2 * x + 1
```

#### Multiprocessing - Parallelization
```python
from multiprocessing import Pool

def process_item(item):
    return item * 2

with Pool() as pool:
    results = pool.map(process_item, data)
```

## Performance Tips

### DO
- Use NumPy for numerical operations
- Vectorize instead of loops
- Use appropriate data structures (set for membership, dict for lookups)
- Profile before optimizing
- Consider Numba for numerical code

### DON'T
- Use Python loops for large numerical arrays
- Concatenate strings in loops (use join)
- Ignore the GIL for CPU-bound tasks
- Optimize prematurely

## Benchmarking

```python
import timeit

# Time a function
time = timeit.timeit(lambda: my_function(), number=1000)
print(f"Time: {time:.6f} seconds")

# Or use time module for quick tests
import time
start = time.time()
my_function()
end = time.time()
print(f"Elapsed: {end - start:.6f} seconds")
```

## Common Patterns

### File I/O
```python
# Always use context managers
with open("file.txt", "r") as f:
    content = f.read()

# Process line by line
with open("file.txt", "r") as f:
    for line in f:
        process(line)
```

### Error Handling
```python
try:
    result = risky_operation()
except FileNotFoundError:
    print("File not found")
except ValueError as e:
    print(f"Invalid value: {e}")
finally:
    cleanup()
```

### List Comprehensions
```python
# Simple and readable
squares = [x**2 for x in range(10)]

# With condition
evens = [x for x in range(10) if x % 2 == 0]

# Use regular loops if too complex
```

## Dependencies

### Core HPC Libraries
```
numpy>=1.21.0
scipy>=1.7.0
numba>=0.54.0
matplotlib>=3.4.0
pandas>=1.3.0
```

Install with:
```bash
pip install -r requirements.txt
```

## Comparing with Other Languages

When implementing algorithms also written in C/C++/CUDA:

### Pure Python
```python
def sum_array(arr):
    total = 0
    for x in arr:
        total += x
    return total
```

### NumPy (Much Faster)
```python
import numpy as np
def sum_array(arr):
    return np.sum(arr)
```

### Numba (Near C Speed)
```python
from numba import jit

@jit(nopython=True)
def sum_array(arr):
    total = 0
    for x in arr:
        total += x
    return total
```

## Learning Path

1. **Master fundamentals (Phase 1-2)**
   - Python syntax and basics
   - Built-in data structures
   - Control flow

2. **Learn OOP (Phase 3)**
   - Critical for larger projects
   - Understand Python object model
   - Magic methods

3. **Advanced features (Phase 4)**
   - Decorators and generators
   - Context managers
   - Iterators

4. **HPC specialization (Phase 5)**
   - NumPy for numerical computing
   - Numba for performance
   - Multiprocessing for parallelism

## Jupyter Notebooks

For interactive exploration:
```bash
# Install Jupyter
pip install jupyter

# Launch notebook
jupyter notebook

# Or use JupyterLab
pip install jupyterlab
jupyter lab
```

## Tips

- Use IPython for enhanced REPL
- Leverage the standard library
- Read the Zen of Python: `import this`
- Use `help()` and `dir()` for exploration
- Write docstrings for functions and classes

## Resources

- Check phase-specific README files (when migrated)
- See `LEARNING_PATH.md` for detailed curriculum
- Refer to `GETTING_STARTED.md` for setup

## Next Steps

- Apply concepts in [projects/](projects/)
- Compare with [C](../c/) and [C++](../cpp/) implementations
- Learn [CUDA](../cuda/) for GPU acceleration
- Use Python for prototyping HPC algorithms

---

**Note**: This directory structure is ready for content migration from existing python-samples repository.
