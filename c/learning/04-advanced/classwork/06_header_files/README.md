# Multi-File C Project: Math Library

This project demonstrates how to create and use **header files** in C for organizing code into multiple files.

## 📁 Project Structure

```
06_header_files/
├── mathlib.h       - Header file (interface/declarations)
├── mathlib.c       - Implementation file (function definitions)
├── main.c          - Main program (uses the library)
├── Makefile        - Build automation
└── README.md       - This file
```

## 🎯 Learning Objectives

- Understand header files and their purpose
- Learn header guards to prevent multiple inclusion
- Separate interface from implementation
- Compile and link multiple source files
- Use Makefile for build automation
- Create reusable code libraries

## 🛠️ How to Build and Run

### Method 1: Using Make (Recommended)

```bash
# Build the program
make

# Run the program
make run

# Clean up compiled files
make clean
```

### Method 2: Manual Compilation (All at Once)

```bash
# Compile and link in one command
gcc -Wall -Wextra main.c mathlib.c -o math_program

# Run
./math_program
```

### Method 3: Separate Compilation (Large Projects)

```bash
# Step 1: Compile each source file to object file
gcc -Wall -Wextra -c mathlib.c -o mathlib.o
gcc -Wall -Wextra -c main.c -o main.o

# Step 2: Link object files
gcc mathlib.o main.o -o math_program

# Step 3: Run
./math_program
```

## 📚 File Descriptions

### mathlib.h (Header File)

The **header file** contains:
- Function declarations (prototypes)
- Type definitions (struct, typedef, enum)
- Macro definitions (#define)
- Constants
- **Header guards** to prevent multiple inclusion

```c
#ifndef MATHLIB_H
#define MATHLIB_H

// Declarations go here

#endif
```

### mathlib.c (Implementation File)

The **implementation file** contains:
- Actual function definitions
- Implements functions declared in the header
- Can have private static functions
- Includes its own header file

### main.c (Application File)

The **main program** contains:
- The main() function
- Uses the library by including mathlib.h
- Calls library functions

## 🔑 Key Concepts

### Header Guards

Prevent a header from being included multiple times:

```c
#ifndef MATHLIB_H    // If not defined
#define MATHLIB_H    // Define it
// ... content ...
#endif               // End if
```

Without header guards, you'd get "redefinition" errors.

### Include Syntax

```c
#include <stdio.h>   // System headers (angle brackets)
#include "mathlib.h" // Your headers (quotes)
```

### Separation of Concerns

- **Header (.h)**: "What" - Interface, what functions are available
- **Source (.c)**: "How" - Implementation, how functions work

## 🎓 Why Use Multiple Files?

1. **Organization**: Related code grouped together
2. **Reusability**: Use library in multiple projects
3. **Maintainability**: Changes in one place
4. **Compilation Speed**: Only recompile changed files
5. **Team Collaboration**: Different people work on different modules
6. **Testing**: Easier to test individual components

## 📋 What the Program Does

The math library provides:
- Basic arithmetic (add, subtract, multiply, divide)
- Advanced math (power, square root, factorial, GCD)
- Prime number checking
- Geometry functions (distance, area)
- Complex number operations

## 🔧 Makefile Targets

```bash
make        # Build the program
make all    # Same as make
make run    # Build and run
make clean  # Remove compiled files
```

## 🎯 Compilation Process

```
Source Files (.c)  →  Compilation  →  Object Files (.o)  →  Linking  →  Executable
     ↓                                      ↓
  mathlib.c  ────────────────────────→  mathlib.o  ──┐
     ↓                                      ↓          │
   main.c    ────────────────────────→   main.o   ────┴──→  math_program
```

## 📝 Best Practices

### For Header Files:
✅ Always use header guards
✅ Only declare, don't define (except inline functions)
✅ Include necessary headers
✅ Document all functions
✅ Use const for parameters that won't change

### For Source Files:
✅ Include corresponding header first
✅ Implement all declared functions
✅ Use static for private helper functions
✅ Add error checking

### For Projects:
✅ One header per module
✅ Match header and source file names
✅ Keep interfaces stable
✅ Minimize dependencies

## 🚀 Exercises

1. **Add new functions** to mathlib:
   - Cube root
   - Logarithm
   - Trigonometric functions

2. **Create new modules**:
   - stringlib (string utilities)
   - filelib (file operations)
   - datastructlib (list, tree, etc.)

3. **Extend existing functions**:
   - Add error codes instead of printing errors
   - Add more complex number operations
   - Add 3D point operations

4. **Practice compilation**:
   - Try all three compilation methods
   - Modify Makefile to add new features
   - Create debug and release builds

5. **Project organization**:
   - Create an include/ directory for headers
   - Create a src/ directory for source files
   - Create a lib/ directory for compiled libraries

## 📖 Further Reading

- [GNU Make Manual](https://www.gnu.org/software/make/manual/)
- [GCC Compilation Process](https://gcc.gnu.org/onlinedocs/)
- [C Header Files Best Practices](https://en.wikipedia.org/wiki/Include_guard)

## 💡 Tips

- If you modify mathlib.h, recompile all files
- If you only modify mathlib.c, only recompile mathlib.c
- Use `-c` flag to create object files without linking
- Use `-o` flag to specify output filename
- Object files (.o) are intermediate files that can be deleted

## 🎉 Success!

If everything compiled and ran correctly, you now understand:
- How to create header files
- How to organize code into multiple files
- How the compilation and linking process works
- How to use Makefiles for building projects

This is a fundamental skill for creating larger C programs and libraries!
