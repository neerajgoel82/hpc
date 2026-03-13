# Phase 1: Foundations - Classwork

This directory contains the learning examples with complete explanations and demonstrations of concepts.

## What's in Classwork?

Classwork files are **teaching materials** that:
- Explain concepts with detailed comments
- Show working examples
- Demonstrate best practices
- Include exercises at the end (to be completed in homework/)

## How to Use

### 1. Read the Code
Open each .c file and read through the comments and code carefully. Understand the concepts being taught.

### 2. Compile and Run
```bash
# From this directory:
make                    # Build all programs
make 01_hello_world     # Build specific program
./01_hello_world        # Run the program
```

### 3. Experiment
Modify the code to test your understanding. Try:
- Changing values
- Adding print statements
- Breaking things to see what happens
- Testing edge cases

### 4. Move to Homework
Once you understand the concepts, go to the homework/ folder to practice.

## Building Programs

```bash
# Build all classwork programs
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

## Files in This Directory

Each .c file focuses on one specific concept. Work through them in order.

## Next Steps

After studying classwork:
1. Review the exercises listed at the end of each file
2. Go to ../homework/ directory
3. Implement solutions to the exercises
4. Test and debug your solutions

## Tips

- Don't just read - type the code yourself
- Use the debugger (lldb/gdb) to step through code
- Compile with warnings: `gcc -Wall -Wextra file.c`
- Ask questions when stuck
- Practice regularly
