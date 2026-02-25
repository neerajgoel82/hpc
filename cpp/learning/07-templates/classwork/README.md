# Module 7: Templates and Generic Programming - Classwork

This directory contains the learning examples with complete explanations and demonstrations of C++ concepts.

## What's in Classwork?

Classwork files are **teaching materials** that:
- Explain C++ concepts with detailed comments
- Show working examples and best practices
- Demonstrate modern C++ features
- Include "TRY THIS" exercises at the end

## How to Use

### 1. Read the Code
Open each .cpp file and read through the comments carefully. These files teach you C++ concepts step by step.

### 2. Compile and Run
```bash
# From this directory:
make                    # Build all programs
make 01_hello_world     # Build specific program
./01_hello_world        # Run the program
```

### 3. Experiment
Modify the code to test your understanding. Try:
- Changing values and observing results
- Adding print statements to trace execution
- Testing edge cases
- Breaking things to understand error messages

### 4. Move to Homework
Once you understand the concepts, go to ../homework/ to practice.

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

## Compilation Flags

All programs are compiled with:
- `-std=c++17`: C++17 standard
- `-Wall -Wextra`: All warnings enabled
- `-O2`: Optimization level 2
- `-g`: Debug symbols

## Files in This Directory

Each .cpp file focuses on specific C++ concepts. Work through them in order for the best learning experience.

## Tips for Learning

- **Type the code**: Don't just read - type examples yourself
- **Use a debugger**: lldb or gdb to step through code
- **Compile often**: Catch errors early
- **Read errors carefully**: C++ error messages can be verbose but informative
- **Experiment freely**: This is your learning space

## Next Steps

After studying classwork:
1. Review the "TRY THIS" sections
2. Go to ../homework/ directory
3. Implement the exercises
4. Test and debug your solutions
