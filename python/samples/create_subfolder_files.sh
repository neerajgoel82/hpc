#!/bin/bash
# Script to create run scripts and READMEs in classwork and homework folders

SAMPLES_DIR="python/samples"

declare -A PHASE_NAMES=(
    ["phase1-foundations"]="Phase 1: Python Foundations"
    ["phase2-intermediate"]="Phase 2: Intermediate Python"
    ["phase3-oop"]="Phase 3: Object-Oriented Programming"
    ["phase4-advanced"]="Phase 4: Advanced Python"
    ["phase5-datascience"]="Phase 5: Data Science with Python"
    ["phase6-pytorch"]="Phase 6: PyTorch Deep Learning"
)

PHASES=(
    "phase1-foundations"
    "phase2-intermediate"
    "phase3-oop"
    "phase4-advanced"
    "phase5-datascience"
    "phase6-pytorch"
)

for phase in "${PHASES[@]}"; do
    phase_name="${PHASE_NAMES[$phase]}"

    echo "Creating files for $phase..."

    # Skip if phase doesn't exist
    if [ ! -d "$SAMPLES_DIR/$phase" ]; then
        echo "  Skipping $phase (not found)"
        continue
    fi

    # ==================== CLASSWORK README ====================
    cat > "$SAMPLES_DIR/$phase/classwork/README.md" << EOF
# $phase_name - Classwork

This directory contains Python learning examples with complete explanations and demonstrations.

## What's in Classwork?

Classwork files are **teaching materials** that:
- Explain Python concepts with detailed comments
- Show working examples and best practices
- Demonstrate Pythonic patterns
- Include exercises (TODO) to practice

## How to Use

### 1. Read the Code
Open each .py file and read through the code and comments carefully.

### 2. Run the Programs
\`\`\`bash
# From this directory:
python3 01_hello_world.py    # Run specific program
\`\`\`

### 3. Experiment
Modify the code to test your understanding. Try:
- Changing values and observing results
- Adding print statements to trace execution
- Testing edge cases
- Breaking things to understand error messages

### 4. Move to Homework
Once you understand the concepts, go to ../homework/ to practice.

## Running Programs

\`\`\`bash
# Run a single program
python3 01_hello_world.py

# Run with Python interpreter
python3 -i 01_hello_world.py    # Interactive mode

# Check syntax
python3 -m py_compile 01_hello_world.py
\`\`\`

## Python Best Practices

- Follow PEP 8 style guide
- Use meaningful variable names
- Add comments for complex logic
- Use type hints (phase 3+)
- Handle exceptions appropriately

## Files in This Directory

Each .py file focuses on specific Python concepts. Work through them in numerical order for the best learning experience.

## Tips for Learning

- **Type the code**: Don't just read - type examples yourself
- **Use REPL**: Test snippets in Python interactive shell
- **Read errors**: Python error messages are helpful
- **Use print()**: Debug by printing values
- **Experiment**: Try variations of the code

## Next Steps

After studying classwork:
1. Review the TODO sections
2. Go to ../homework/ directory
3. Implement the exercises
4. Test and debug your solutions

## Virtual Environment (Optional)

\`\`\`bash
# Create virtual environment
python3 -m venv .venv

# Activate
source .venv/bin/activate  # macOS/Linux
.venv\\Scripts\\activate   # Windows

# Install dependencies (if any)
pip install -r requirements.txt
\`\`\`
EOF

    # ==================== HOMEWORK README ====================
    cat > "$SAMPLES_DIR/$phase/homework/README.md" << EOF
# $phase_name - Homework

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
\`\`\`python
# Your solution here
def solution():
    pass
\`\`\`

### 4. Run and Test
\`\`\`bash
# From this directory:
python3 01_hello_world.py    # Run your homework
\`\`\`

### 5. Debug and Iterate
If something doesn't work:
- Read Python error messages carefully (they're helpful!)
- Use print() for debugging
- Use Python debugger (pdb)
- Review the classwork file
- Test with different inputs

## Running Programs

\`\`\`bash
# Run homework
python3 01_hello_world.py

# Run with debugging
python3 -m pdb 01_hello_world.py

# Check syntax
python3 -m py_compile 01_hello_world.py
\`\`\`

## Workflow

1. **Study**: Read ../classwork/01_hello_world.py
2. **Practice**: Edit homework/01_hello_world.py
3. **Test**: \`python3 01_hello_world.py\`
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
\`\`\`bash
# Common fixes:
# - Missing colon after if/for/def
# - Indentation errors (use 4 spaces)
# - Unmatched parentheses/brackets
# - Missing quotes around strings
\`\`\`

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

\`\`\`python
# Add print statements
print(f"Debug: variable = {variable}")

# Use breakpoint() (Python 3.7+)
breakpoint()  # Starts debugger

# Use pdb
import pdb
pdb.set_trace()  # Older way
\`\`\`

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
EOF

    echo "  ✓ Created README in classwork/"
    echo "  ✓ Created README in homework/"
done

echo ""
echo "All READMEs created successfully!"
