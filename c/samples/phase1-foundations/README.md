# Phase 1: Foundations

**Duration**: Weeks 1-3 (3 weeks)

**Goal**: Master the fundamental building blocks of C programming.

---

## What You'll Learn

- Basic syntax and program structure
- Data types and variables
- Operators (arithmetic, logical, bitwise)
- Control flow (if/else, switch, loops)
- Input and output operations
- How C programs are compiled and executed

---

## Programs in This Phase

### 01. Hello World (`01_hello_world.c`)
**Concept**: Basic program structure, `printf()`

Your first C program! Learn about:
- The `main()` function
- Standard output with `printf()`
- Return values
- Compilation basics

**Compile & Run**:
```bash
gcc -std=c11 -Wall -Wextra 01_hello_world.c -o hello_world
./hello_world
```

---

### 02. Data Types (`02_data_types.c`)
**Concept**: Primitive types, sizes, ranges

Explore C's data types:
- Integer types: `int`, `short`, `long`, `long long`
- Character type: `char`
- Floating-point: `float`, `double`
- Type sizes with `sizeof()`
- Type limits and ranges

**Key Takeaway**: Understanding data type sizes is crucial for memory efficiency in HPC.

---

### 03. Operators (`03_operators.c`)
**Concept**: Arithmetic, comparison, logical operators

Master all operator types:
- Arithmetic: `+`, `-`, `*`, `/`, `%`
- Comparison: `==`, `!=`, `<`, `>`, `<=`, `>=`
- Logical: `&&`, `||`, `!`
- Increment/decrement: `++`, `--`
- Assignment: `=`, `+=`, `-=`, etc.

---

### 04. Input/Output (`04_input_output.c`)
**Concept**: Reading input, formatted output

Learn I/O operations:
- `scanf()` for reading input
- `printf()` format specifiers (`%d`, `%f`, `%c`, `%s`)
- Buffer handling
- Common input/output patterns

---

### 05. If/Else Statements (`05_if_else.c`)
**Concept**: Conditional execution

Control program flow with:
- `if` statements
- `if-else` chains
- Nested conditionals
- Ternary operator `? :`
- Common conditional patterns

---

### 06. Switch Statement (`06_switch.c`)
**Concept**: Multi-way branching

Learn the switch statement:
- `switch-case` syntax
- `break` statements
- `default` case
- When to use switch vs if/else
- Fall-through behavior

---

### 07. Loops (`07_loops.c`)
**Concept**: Iteration and repetition

Master all loop types:
- `for` loops (counting, iteration)
- `while` loops (condition-based)
- `do-while` loops (execute at least once)
- Nested loops
- `break` and `continue` statements
- Infinite loops and exit conditions

---

### 08. Bitwise Operators (`08_bitwise_operators.c`)
**Concept**: Bit-level manipulation

Low-level operations on bits:
- Bitwise AND (`&`), OR (`|`), XOR (`^`)
- NOT (`~`)
- Left shift (`<<`), right shift (`>>`)
- Practical uses: flags, masks, optimization
- Bit manipulation tricks

**Why It Matters**: Critical for systems programming and performance optimization.

---

## Quick Start

### Build All Programs
```bash
make
```

### Build Specific Program
```bash
make hello_world
# or
gcc -std=c11 -Wall -Wextra 01_hello_world.c -o hello_world
```

### Run a Program
```bash
./hello_world
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

**Flags Explained**:
- `-std=c11` - Use C11 standard
- `-Wall` - Enable all warnings
- `-Wextra` - Enable extra warnings
- `-o program` - Output executable name

### With Debugging
```bash
gcc -std=c11 -Wall -Wextra -g program.c -o program
```
- `-g` - Include debugging symbols for gdb

### With Optimization
```bash
gcc -std=c11 -Wall -Wextra -O2 program.c -o program
```
- `-O2` - Level 2 optimization

---

## Learning Checklist

After completing Phase 1, you should be able to:

- [ ] Write and compile a basic C program
- [ ] Choose appropriate data types for variables
- [ ] Use arithmetic and logical operators correctly
- [ ] Read input and produce formatted output
- [ ] Write conditional statements (if/else, switch)
- [ ] Use all three types of loops appropriately
- [ ] Understand and use bitwise operators
- [ ] Read and understand compiler warnings
- [ ] Debug simple programs with print statements
- [ ] Understand how C programs are executed

---

## Common Mistakes to Avoid

1. **Forgetting semicolons** - Every statement needs one
2. **Using `=` instead of `==`** - Assignment vs comparison
3. **Integer division surprises** - `5/2` = `2`, not `2.5`
4. **Uninitialized variables** - Always initialize before use
5. **Missing `break` in switch** - Can cause fall-through bugs
6. **Infinite loops** - Make sure loop conditions can become false
7. **Ignoring compiler warnings** - They're there for a reason!

---

## Practice Exercises

1. **Calculator**: Build a simple calculator using switch statement
2. **Temperature Converter**: Fahrenheit ↔ Celsius with input validation
3. **Prime Checker**: Use loops to check if a number is prime
4. **Bit Counter**: Count number of set bits (1s) in an integer
5. **Pattern Printer**: Use nested loops to print various patterns

---

## Next Steps

Once comfortable with Phase 1:
- **Move to Phase 2**: Functions and program organization
- **Practice**: Complete all exercises above
- **Review**: Make sure you understand every concept
- **Experiment**: Modify programs to see what happens

---

## Tips for Success

- **Type every program yourself** - Don't copy-paste
- **Predict before running** - What will the output be?
- **Make deliberate mistakes** - Learn from compiler errors
- **Use the debugger** - Learn to use gdb early
- **Read compiler warnings** - They teach you proper C
- **Experiment freely** - Break things to understand them

---

## Resources

- [C Reference](https://en.cppreference.com/w/c)
- [GCC Documentation](https://gcc.gnu.org/onlinedocs/)
- Main curriculum: [../../README.md](../../README.md)

---

**Time Estimate**: 3 weeks at 5-7 hours/week

**Prerequisite**: None - start here!

**Next Phase**: [Phase 2: Building Blocks](../phase2-building-blocks/)
