# Phase 5: Real-World Projects - Homework

This directory contains exercise files for practicing the concepts learned in classwork.

## What's in Homework?

Homework files are **practice exercises** that:
- List exercises from the corresponding classwork file
- Provide a template structure
- Give you space to implement solutions
- Test your understanding of concepts

## How to Use

### 1. Read Classwork First
Always read and understand the corresponding classwork file before attempting homework.

### 2. Read the Exercises
Each homework file contains exercises extracted from the classwork file. Read them carefully.

### 3. Implement Solutions
Write your code in the homework file. Follow the structure provided:
```c
int main() {
    // Your implementation here
    return 0;
}
```

### 4. Compile and Test
```bash
# From this directory:
make                    # Build all homework
make 01_hello_world     # Build specific homework
./01_hello_world        # Run and test your solution
```

### 5. Debug and Iterate
If something doesn't work:
- Read the error messages carefully
- Use printf() debugging
- Use a debugger (lldb/gdb)
- Review the classwork file
- Test with different inputs

## Building Programs

```bash
# Build all homework programs
make

# Build specific program
make 01_hello_world

# Clean executables
make clean

# List all programs
make list

# Get help
make help
```

## Workflow

1. **Study**: Read ../classwork/01_hello_world.c
2. **Practice**: Edit homework/01_hello_world.c
3. **Compile**: `make 01_hello_world`
4. **Test**: `./01_hello_world`
5. **Iterate**: Debug and improve

## Tips for Success

- **Start Simple**: Begin with the first exercise
- **Test Often**: Compile and run after each change
- **Use Comments**: Document your approach
- **Handle Errors**: Check return values and inputs
- **Be Patient**: Debugging is part of learning
- **Ask Questions**: When truly stuck, seek help

## Common Issues

### Compilation Errors
```bash
# Read error messages carefully
gcc -std=c11 -Wall -Wextra 01_hello_world.c -o 01_hello_world

# Common fixes:
# - Missing semicolon
# - Undeclared variable
# - Type mismatch
# - Missing header file
```

### Runtime Errors
- Use printf() to debug
- Check array bounds
- Verify pointer validity
- Test with simple inputs first

## Completion Checklist

Mark exercises as complete when:
- [ ] Code compiles without warnings
- [ ] Program runs correctly
- [ ] Edge cases handled
- [ ] Code is readable and commented
- [ ] You understand why it works

## Next Steps

After completing homework:
1. Compare with classwork approach
2. Optimize if needed
3. Move to next file
4. Review periodically
