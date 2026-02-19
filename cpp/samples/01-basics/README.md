# Module 1: C++ Fundamentals

## Overview
This module covers the absolute fundamentals of C++ programming. Master these concepts before moving forward - they're the foundation for everything else.

## Compilation Quick Reference

```bash
# Compile a single file
g++ -std=c++17 -o program_name source_file.cpp

# Run the compiled program
./program_name

# Compile with warnings (recommended!)
g++ -std=c++17 -Wall -Wextra -o program_name source_file.cpp
```

## VS Code Quick Start
1. Open any `.cpp` file
2. Press `Cmd+Shift+B` → "Build and Run C++ (current file)"
3. Press `F5` to debug with breakpoints

## Files in This Module

1. **01_hello_world.cpp** - Your first C++ program
2. **02_types_and_variables.cpp** - Data types and sizes (critical for GPU)
3. **03_functions.cpp** - Creating and calling functions
4. **04_control_flow.cpp** - If/else, loops, switch statements
5. **05_strings.cpp** - Working with std::string
6. **06_input.cpp** - Getting user input with cin
7. **07_enums.cpp** - Enumerations (enum and enum class)
8. **08_const_constexpr.cpp** - Constants and compile-time evaluation
9. **09_namespaces.cpp** - Code organization with namespaces

## Learning Approach

For each file:
1. **Read** the code and comments carefully
2. **Predict** what the output will be
3. **Compile and run** using VS Code or command line
4. **Verify** your prediction matches the output
5. **Experiment** with the "TRY THIS" suggestions
6. **Break it** - try modifying code to see what errors you get

## Key Concepts

### Data Types
- Understanding `int`, `float`, `double`, `bool`, `char`
- Why type sizes matter (especially for GPU)
- Fixed-width types (`int32_t`, `int64_t`)

### Control Flow
- Making decisions with `if/else` and `switch`
- Loops: `for`, `while`, `do-while`
- `break` and `continue`

### Functions
- Declaring and defining functions
- Parameters and return values
- Function overloading

### Strings
- `std::string` vs C-strings
- String operations and methods
- Converting between strings and numbers

### Enumerations
- Traditional `enum` vs `enum class`
- When and why to use enums
- Type safety with `enum class`

### Constants
- `const` for runtime constants
- `constexpr` for compile-time constants
- Using constants effectively

### Namespaces
- Organizing code
- Avoiding name conflicts
- Using `std::` and custom namespaces

## Common Mistakes to Avoid

1. **Forgetting semicolons** - Every statement needs one
2. **Uninitialized variables** - Always initialize: `int x = 0;`
3. **Using `=` instead of `==`** - Assignment vs comparison
4. **Off-by-one errors in loops** - Watch your loop bounds
5. **Integer division** - `5/2` gives `2`, not `2.5`

## Practice Exercises

After completing all files, try these:

1. **Temperature Converter**: Convert Celsius ↔ Fahrenheit
2. **Prime Checker**: Check if a number is prime
3. **Fibonacci Generator**: Print first N Fibonacci numbers
4. **Simple Calculator**: Ask for two numbers and an operation
5. **Grade Calculator**: Input scores, calculate and display average

## GPU Programming Relevance

| Concept | GPU Relevance |
|---------|---------------|
| Data types | GPUs care deeply about float vs double performance |
| Control flow | Loops process arrays; conditional logic in kernels |
| Functions | GPU kernels are like functions that run in parallel |
| Const/constexpr | Block sizes, grid dimensions often compile-time constants |
| Enums | Memory types, error codes in CUDA use enums |

## Next Steps

After completing Module 1, you'll have a solid foundation in C++ basics.

**Module 2** covers functions in depth, header files, and the preprocessor - essential for organizing real projects and understanding how CUDA code is structured.

## Troubleshooting

**Compilation errors?**
- Check for missing semicolons
- Ensure all brackets `{}` and parentheses `()` are balanced
- Check if you included necessary headers (`<iostream>`, `<string>`)

**Program crashes?**
- Check for division by zero
- Ensure array indices are within bounds
- Initialize all variables before use

**Unexpected output?**
- Add `std::cout` statements to debug
- Use VS Code debugger (F5) to step through code
- Check operator precedence (use parentheses when unsure)

## Estimated Time
- **Reading all examples**: 2-3 hours
- **Experimenting with "TRY THIS"**: 3-5 hours
- **Practice exercises**: 3-5 hours
- **Total**: 8-13 hours

Take your time - understanding these fundamentals is crucial!
