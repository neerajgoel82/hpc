# Phase 2: Building Blocks

**Duration**: Weeks 4-6 (3 weeks)

**Goal**: Learn to organize code with functions and work with arrays and strings.

---

## What You'll Learn

- Function definition and declaration
- Function parameters and return values
- Variable scope and lifetime
- Recursion
- Arrays (1D and multidimensional)
- String handling
- Command-line arguments
- Program organization

---

## Programs in This Phase

### 01. Functions (`01_functions.c`)
**Concept**: Function basics, parameters, return values

Learn to write and use functions:
- Function declaration vs definition
- Parameters and arguments
- Return values and `void` functions
- Function prototypes
- Why functions are essential

**Key Takeaway**: Functions enable code reuse and modular design.

**Compile & Run**:
```bash
gcc -std=c11 -Wall -Wextra 01_functions.c -o functions
./functions
```

---

### 02. Function Scope (`02_function_scope.c`)
**Concept**: Variable scope, lifetime, storage classes

Understanding scope and visibility:
- Local vs global variables
- Function scope
- Block scope
- Storage classes: `auto`, `static`, `extern`
- Variable lifetime
- Name shadowing

**Key Takeaway**: Proper scope management prevents bugs and improves code clarity.

---

### 03. Recursion (`03_recursion.c`)
**Concept**: Recursive function calls

Master recursion:
- Base case and recursive case
- Call stack behavior
- Classic examples: factorial, Fibonacci
- When to use recursion vs iteration
- Tail recursion
- Stack overflow considerations

**Key Takeaway**: Recursion is powerful but use carefully in HPC due to stack limitations.

---

### 04. Arrays (`04_arrays.c`)
**Concept**: Array declaration, initialization, access

Working with arrays:
- Array declaration and initialization
- Zero-based indexing
- Array bounds (no automatic checking!)
- Passing arrays to functions
- Array sizes with `sizeof()`
- Common array operations

**Key Takeaway**: Arrays are the foundation of data processing in C.

---

### 05. Multidimensional Arrays (`05_multidimensional_arrays.c`)
**Concept**: 2D arrays, matrices, memory layout

Understanding multidimensional data:
- 2D array declaration
- Row-major order (C's default)
- Matrix operations
- Passing 2D arrays to functions
- Memory layout visualization
- 3D arrays and beyond

**Key Takeaway**: Memory layout matters for cache performance in HPC.

---

### 06. Strings (`06_strings.c`)
**Concept**: Character arrays, string handling

C strings in depth:
- Strings as character arrays
- Null terminator `\0`
- String literals
- String library functions (`strlen`, `strcpy`, `strcmp`, etc.)
- String input/output
- Common string operations
- Buffer overflow dangers

**Key Takeaway**: C strings require careful handling - no bounds checking!

---

### 07. Command-Line Arguments (`07_command_line_args.c`)
**Concept**: `argc` and `argv`, processing arguments

Making programs interactive:
- `int main(int argc, char *argv[])`
- Argument count (`argc`)
- Argument vector (`argv`)
- Processing command-line options
- Argument validation
- Practical examples

**Key Takeaway**: Essential for building real command-line tools.

---

## Quick Start

### Build All Programs
```bash
make
```

### Build Specific Program
```bash
make functions
# or
gcc -std=c11 -Wall -Wextra 01_functions.c -o functions
```

### Run a Program
```bash
./functions
```

### Test Command-Line Arguments
```bash
./command_line_args arg1 arg2 arg3
```

### Build and Test All
```bash
make test
```

### Clean Up
```bash
make clean
```

---

## Compilation Guide

### Standard Compilation
```bash
gcc -std=c11 -Wall -Wextra program.c -o program
```

### With Math Library (for recursion examples)
```bash
gcc -std=c11 -Wall -Wextra program.c -o program -lm
```

### Multiple Source Files (introduced in this phase)
```bash
gcc -std=c11 -Wall -Wextra file1.c file2.c -o program
```

---

## Learning Checklist

After completing Phase 2, you should be able to:

- [ ] Write functions with parameters and return values
- [ ] Understand function prototypes and why they're needed
- [ ] Explain variable scope and lifetime
- [ ] Write recursive functions correctly
- [ ] Declare and initialize arrays
- [ ] Pass arrays to functions
- [ ] Work with 2D arrays (matrices)
- [ ] Manipulate C strings safely
- [ ] Use string library functions
- [ ] Process command-line arguments
- [ ] Understand memory layout of arrays
- [ ] Organize code into multiple functions

---

## Common Mistakes to Avoid

1. **Array out of bounds** - C doesn't check indices!
2. **Forgetting null terminator** - Strings must end with `\0`
3. **Returning local array address** - Arrays are not returned by value
4. **Missing base case in recursion** - Causes infinite recursion
5. **Confusing `argc` and `argv`** - `argc` is count, `argv` is array
6. **Using uninitialized arrays** - Garbage values!
7. **Buffer overflows** - Always check string lengths
8. **Modifying string literals** - They're read-only!

---

## Practice Exercises

1. **Array Statistics**: Write functions for min, max, average of an array
2. **Matrix Operations**: Implement matrix addition and multiplication
3. **String Reverser**: Reverse a string in-place
4. **Word Counter**: Count words in a string
5. **Tower of Hanoi**: Implement the classic recursion puzzle
6. **File Processor**: Program that takes filename as command-line argument
7. **Bubble Sort**: Sort an array using nested loops

---

## Key Concepts for HPC

### Arrays and Memory
- Arrays are stored contiguously in memory
- Cache-friendly access: iterate in order
- 2D arrays: row-major order in C (access row-by-row for best performance)

### Functions and Performance
- Function call overhead (usually negligible with optimization)
- Inline functions for very small, frequently-called functions
- Recursion can be expensive (stack overhead)

### Why This Matters
- Arrays are the building block of matrices and tensors
- Understanding memory layout crucial for GPU programming
- Function organization essential for large HPC codes

---

## Code Organization Tips

### Good Function Design
```c
// Good: Single responsibility
int sumArray(int arr[], int size);

// Good: Clear names
double calculateAverage(double values[], int count);

// Bad: Does too much
int processEverything(int arr[], int size);
```

### Array Best Practices
```c
// Good: Pass size explicitly
void processArray(int arr[], int size);

// Good: Use const for read-only arrays
void printArray(const int arr[], int size);

// Bad: Assuming array size
void badFunction(int arr[]);  // Size unknown!
```

---

## Next Steps

Once comfortable with Phase 2:
- **Move to Phase 3**: Pointers and dynamic memory
- **Practice**: Complete all exercises above
- **Review Phase 1**: Make sure basics are solid
- **Build a mini-project**: Combine Phase 1 & 2 concepts

---

## Mini-Project Ideas

1. **Student Grade Manager**
   - Array of student scores
   - Functions to calculate average, min, max
   - Command-line input for number of students

2. **Text Adventure Game**
   - Use strings for commands
   - Functions for different game actions
   - Array of room descriptions

3. **Simple Matrix Calculator**
   - 2D arrays for matrices
   - Functions for add, multiply, transpose
   - Command-line arguments for operation choice

---

## Resources

- [C Reference - Functions](https://en.cppreference.com/w/c/language/functions)
- [C Reference - Arrays](https://en.cppreference.com/w/c/language/array)
- [C Reference - String Functions](https://en.cppreference.com/w/c/string/byte)
- Main curriculum: [../../README.md](../../README.md)

---

## Tips for Success

- **Draw the call stack** - Visualize function calls and returns
- **Draw array memory layouts** - Understand how data is stored
- **Use debugger** - Step through recursive calls with gdb
- **Test boundary conditions** - First/last array elements, empty arrays
- **Validate string lengths** - Always check before operations
- **Use `const` correctly** - Prevents accidental modification
- **Comment function contracts** - Document parameters and return values

---

**Time Estimate**: 3 weeks at 6-8 hours/week

**Prerequisite**: Phase 1 complete

**Next Phase**: [Phase 3: Core Concepts](../phase3-core-concepts/)

**Previous Phase**: [Phase 1: Foundations](../phase1-foundations/)
