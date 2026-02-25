# Phase 6: PyTorch Deep Learning - Homework

This directory contains exercise files for practicing Python concepts learned in classwork.

## What's in Homework?

Homework files are **practice exercises** that:
- List TODO/exercises from classwork
- Provide a template structure
- Give you space to implement solutions
- Test your understanding of Python concepts

## How to Use

### 1. Read Classwork First
Always read and understand the corresponding classwork file before attempting homework.

### 2. Read the Exercises
Each homework file contains exercises extracted from the classwork file.

### 3. Implement Solutions
Write your Python code in the homework file:
```python
# Your solution here
def solution():
    pass
```

### 4. Run and Test
```bash
# From this directory:
python3 01_hello_world.py    # Run your homework
```

### 5. Debug and Iterate
If something doesn't work:
- Read Python error messages carefully (they're helpful!)
- Use print() for debugging
- Use Python debugger (pdb)
- Review the classwork file
- Test with different inputs

## Running Programs

```bash
# Run homework
python3 01_hello_world.py

# Run with debugging
python3 -m pdb 01_hello_world.py

# Check syntax
python3 -m py_compile 01_hello_world.py
```

## Workflow

1. **Study**: Read ../classwork/01_hello_world.py
2. **Practice**: Edit homework/01_hello_world.py
3. **Test**: `python3 01_hello_world.py`
4. **Iterate**: Debug and improve

## Tips for Success

- **Start Simple**: Begin with the first TODO
- **Test Often**: Run after each change
- **Use REPL**: Test snippets interactively
- **Read Errors**: Python errors tell you what's wrong
- **Use print()**: Debug by printing values
- **Be Patient**: Python is learner-friendly

## Common Issues

### Syntax Errors
```bash
# Common fixes:
# - Missing colon after if/for/def
# - Indentation errors (use 4 spaces)
# - Unmatched parentheses/brackets
# - Missing quotes around strings
```

### Runtime Errors
- Check variable names (typos)
- Verify data types
- Check list indices (0-indexed)
- Handle None values
- Test with simple inputs first

### Logic Errors
- Use print() to debug
- Check loop conditions
- Verify function return values
- Test edge cases

## Python Debugging

```python
# Add print statements
print(f"Debug: variable = {variable}")

# Use breakpoint() (Python 3.7+)
breakpoint()  # Starts debugger

# Use pdb
import pdb
pdb.set_trace()  # Older way
```

## Code Style

- Follow PEP 8
- Use 4 spaces for indentation
- Maximum line length: 79-88 characters
- Use meaningful variable names
- Add docstrings to functions (phase 3+)

## Completion Checklist

Mark exercises as complete when:
- [ ] Code runs without errors
- [ ] All TODO items implemented
- [ ] Edge cases handled
- [ ] Code follows Python style guide
- [ ] You understand why it works

## Next Steps

After completing homework:
1. Compare with classwork approach
2. Learn Pythonic alternatives
3. Move to next file
4. Review periodically

## Resources

- Python Documentation: https://docs.python.org
- PEP 8 Style Guide: https://pep8.org
- Real Python: https://realpython.com
- Python Tutor: https://pythontutor.com (visualize code execution)
