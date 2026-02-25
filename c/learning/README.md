# C Programming for Systems and HPC

A comprehensive, hands-on curriculum covering C programming from fundamentals through advanced systems programming. This repository provides 26 structured programs across 5 phases designed to take you from absolute beginner to intermediate C programmer with a focus on high-performance computing.

## What You'll Learn

- **C Fundamentals** - Syntax, data types, operators, control flow
- **Functions and Arrays** - Modular programming, collections, strings
- **Pointers and Memory** - Memory addresses, pointer arithmetic, dynamic allocation
- **Data Structures** - Custom types with structures, linked lists, stacks
- **File I/O** - Reading and writing files, persistence
- **Systems Programming** - Memory management, preprocessor, low-level operations
- **Complete Projects** - Real applications combining all concepts

---

## Quick Start

### Prerequisites
- macOS, Linux, or Windows with WSL
- GCC compiler with C11 support
- Text editor or IDE (VS Code/Cursor recommended)

### Verify Your Setup
```bash
gcc --version
```

You should see GCC version information. If not, install GCC:

**macOS:**
```bash
xcode-select --install
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install build-essential
```

**Windows:**
Install MinGW or use WSL (Windows Subsystem for Linux)

### Your First Program
```bash
# Navigate to C learning directory
cd c/learning

# Compile and run your first program
gcc -std=c11 -Wall -Wextra phase1-foundations/01_hello_world.c -o hello
./hello
```

### Using VS Code/Cursor
1. Open any `.c` file
2. Press `Cmd+Shift+B` (or `Ctrl+Shift+B` on Windows/Linux)
3. Select "Build and Run"
4. Press `F5` to debug with breakpoints

### Using the Makefile
```bash
# Compile all Phase 1 programs
make phase1

# Run a specific program
make run FILE=phase1-foundations/01_hello_world.c

# Compile everything
make all

# Clean up compiled files
make clean
```

---

## Repository Structure

```
c/learning/
├── phase1-foundations/        # 7 files - Weeks 1-2
│   ├── 01_hello_world.c
│   ├── 02_data_types.c
│   ├── 03_operators.c
│   ├── 04_input_output.c
│   ├── 05_if_else.c
│   ├── 06_switch.c
│   └── 07_loops.c
│
├── phase2-building-blocks/    # 6 files - Weeks 3-5
│   ├── 01_functions.c
│   ├── 02_function_scope.c
│   ├── 03_recursion.c
│   ├── 04_arrays.c
│   ├── 05_multidimensional_arrays.c
│   └── 06_strings.c
│
├── phase3-core-concepts/      # 5 files - Weeks 6-9
│   ├── 01_pointers_basics.c
│   ├── 02_pointers_arrays.c
│   ├── 03_pointers_functions.c
│   ├── 04_structures.c
│   └── 05_dynamic_memory.c
│
├── phase4-advanced/           # 4+ files - Weeks 10-14
│   ├── 01_file_io.c
│   ├── 02_preprocessor.c
│   ├── 03_linked_list.c
│   ├── 04_stack.c
│   └── 06_header_files/       # Multi-file project
│
├── phase5-projects/           # 2 files - Weeks 15-16
│   ├── 01_contact_manager.c
│   └── 02_text_analyzer.c
│
├── README.md                  # This file
├── Makefile                   # Build automation
├── test_compilation.sh        # Test script
└── .gitignore                 # Git configuration
```

---

## Complete Curriculum (5 Phases, 26 Programs)

### Phase 1: Foundations (Weeks 1-2)

**Goal**: Master the absolute basics of C programming

| File | Topic | Key Concepts |
|------|-------|--------------|
| 01_hello_world.c | First Program | Basic structure, main(), printf(), return |
| 02_data_types.c | Variables & Types | int, float, double, char, sizeof |
| 03_operators.c | Operations | Arithmetic, relational, logical, increment |
| 04_input_output.c | User Interaction | scanf(), printf(), format specifiers |
| 05_if_else.c | Decisions | if-else, conditions, ternary operator |
| 06_switch.c | Multiple Choices | switch-case, break, default |
| 07_loops.c | Repetition | while, do-while, for, break, continue |

**What You'll Build**:
- Simple calculator
- Number guessing game
- Pattern printing programs

**Skills Gained**:
- Write, compile, and run C programs
- Understand and use all basic data types
- Work with operators and expressions
- Handle user input and output
- Implement conditional logic
- Use all types of loops effectively

**Estimated time**: 10-15 hours (1-2 weeks)

---

### Phase 2: Building Blocks (Weeks 3-5)

**Goal**: Write modular, reusable code with functions and arrays

| File | Topic | Key Concepts |
|------|-------|--------------|
| 01_functions.c | Reusable Code | Function declaration, parameters, return values |
| 02_function_scope.c | Variable Lifetime | Local, global, static variables |
| 03_recursion.c | Self-Calling Functions | Base case, recursive case, call stack |
| 04_arrays.c | Collections | Declaration, initialization, iteration |
| 05_multidimensional_arrays.c | Matrices | 2D arrays, nested loops, matrix operations |
| 06_strings.c | Text Processing | Character arrays, string functions, null terminator |

**What You'll Build**:
- Sorting and searching programs
- Matrix calculator
- Text manipulation tools

**Skills Gained**:
- Write and use functions effectively
- Understand variable scope and lifetime
- Implement recursive algorithms
- Work with arrays and strings
- Process multi-dimensional data

**Estimated time**: 15-20 hours (2-3 weeks)

---

### Phase 3: Core Concepts (Weeks 6-9)

**Goal**: Master pointers and dynamic memory - the heart of C

| File | Topic | Key Concepts |
|------|-------|--------------|
| 01_pointers_basics.c | Memory Addresses | Pointers, &, *, dereferencing, NULL |
| 02_pointers_arrays.c | Pointer Arithmetic | Array-pointer relationship, traversal |
| 03_pointers_functions.c | Advanced Pointers | Pass by reference, function pointers |
| 04_structures.c | Custom Types | struct, typedef, nested structures |
| 05_dynamic_memory.c | Runtime Allocation | malloc, calloc, realloc, free |

**What You'll Build**:
- Dynamic data structures
- Custom type systems
- Memory-efficient programs

**Skills Gained**:
- Master pointers and memory addresses
- Perform pointer arithmetic
- Use function pointers
- Create custom data types with structures
- Manage dynamic memory allocation
- Prevent memory leaks

**Estimated time**: 15-20 hours (3-4 weeks)

---

### Phase 4: Advanced Topics (Weeks 10-14)

**Goal**: File handling, preprocessor, and data structures

| File | Topic | Key Concepts |
|------|-------|--------------|
| 01_file_io.c | File Operations | fopen, fclose, fprintf, fscanf, binary I/O |
| 02_preprocessor.c | Compile-Time Features | #define, #include, conditional compilation |
| 03_linked_list.c | Dynamic Lists | Nodes, insertion, deletion, traversal |
| 04_stack.c | LIFO Structure | Push, pop, peek, applications |
| 06_header_files/ | Multi-File Projects | Header files, compilation, linking |

**What You'll Build**:
- File processing tools
- Data structure implementations
- Expression evaluators
- Modular library code

**Skills Gained**:
- Read from and write to files
- Use preprocessor directives
- Implement linked lists
- Build stack data structures
- Organize code across multiple files
- Debug complex programs
- Write maintainable code

**Estimated time**: 12-16 hours (3-4 weeks)

---

### Phase 5: Real Projects (Weeks 15-16)

**Goal**: Apply everything in complete, practical applications

| File | Description | Concepts Used |
|------|-------------|---------------|
| 01_contact_manager.c | Full CRUD app | Structures, dynamic memory, file I/O, menu system |
| 02_text_analyzer.c | File analysis tool | File I/O, string processing, algorithms, statistics |

**What You'll Build**:
- Production-ready applications
- Complete systems with persistence
- Professional coding practices

**Skills Gained**:
- Design complete applications
- Implement CRUD operations
- Handle errors gracefully
- Write user-friendly interfaces
- Persist data to files
- Apply best practices

**Estimated time**: 10-15 hours (2 weeks)

---

## Compilation Guide

### Standard Compilation
```bash
# C11 standard with all warnings
gcc -std=c11 -Wall -Wextra file.c -o output

# With debugging symbols
gcc -std=c11 -Wall -Wextra -g file.c -o output

# With optimization
gcc -std=c11 -Wall -Wextra -O2 file.c -o output

# Run the program
./output
```

### Compilation Flags Explained

| Flag | Purpose |
|------|---------|
| `-std=c11` | Use C11 standard |
| `-Wall` | Enable all warnings |
| `-Wextra` | Enable extra warnings |
| `-g` | Include debugging information |
| `-O2` | Optimize for performance |
| `-o output` | Specify output filename |

### Multi-File Compilation

**Method 1: Compile and link at once**
```bash
gcc -Wall -Wextra main.c mathlib.c -o program
```

**Method 2: Separate compilation (for large projects)**
```bash
# Step 1: Compile to object files
gcc -Wall -Wextra -c mathlib.c -o mathlib.o
gcc -Wall -Wextra -c main.c -o main.o

# Step 2: Link object files
gcc mathlib.o main.o -o program
```

### Debugging with GDB
```bash
# Compile with debug symbols
gcc -std=c11 -Wall -g program.c -o program

# Run with debugger
gdb ./program

# Common GDB commands:
# run          - Start the program
# break main   - Set breakpoint at main
# next         - Execute next line
# step         - Step into function
# print var    - Print variable value
# quit         - Exit GDB
```

### Checking for Memory Leaks (Linux/macOS)
```bash
# Install valgrind if needed
# macOS: brew install valgrind
# Linux: sudo apt-get install valgrind

# Run with valgrind
valgrind --leak-check=full ./program
```

---

## Learning Paths

### Path 1: Standard Track (16 weeks)
**4-6 hours per week, steady pace**

- **Weeks 1-2**: Phase 1 (1 file per day)
- **Weeks 3-5**: Phase 2 (1 file per day)
- **Weeks 6-9**: Phase 3 (slower pace for pointers)
- **Weeks 10-14**: Phase 4 (1 file per day)
- **Weeks 15-16**: Phase 5 (1 project per week)

**Best for**: Working professionals, students with other courses

---

### Path 2: Intensive Track (8 weeks)
**8-12 hours per week, fast pace**

- **Week 1**: Phase 1 (all files)
- **Weeks 2-3**: Phase 2
- **Weeks 4-5**: Phase 3
- **Weeks 6-7**: Phase 4
- **Week 8**: Phase 5

**Best for**: Bootcamp students, career changers, full-time learners

---

### Path 3: Relaxed Track (24+ weeks)
**2-3 hours per week, your own pace**

- Take your time with each concept
- Perfect for busy schedules
- Focus on deep understanding

**Best for**: Hobbyists, self-learners, evenings/weekends

---

## Module Completion Checklist

Track your progress through the curriculum:

```
[ ] Phase 1: Foundations (7/7 files)
    [ ] 01_hello_world.c
    [ ] 02_data_types.c
    [ ] 03_operators.c
    [ ] 04_input_output.c
    [ ] 05_if_else.c
    [ ] 06_switch.c
    [ ] 07_loops.c

[ ] Phase 2: Building Blocks (6/6 files)
    [ ] 01_functions.c
    [ ] 02_function_scope.c
    [ ] 03_recursion.c
    [ ] 04_arrays.c
    [ ] 05_multidimensional_arrays.c
    [ ] 06_strings.c

[ ] Phase 3: Core Concepts (5/5 files)
    [ ] 01_pointers_basics.c
    [ ] 02_pointers_arrays.c
    [ ] 03_pointers_functions.c
    [ ] 04_structures.c
    [ ] 05_dynamic_memory.c

[ ] Phase 4: Advanced Topics (4/4 files + 1 project)
    [ ] 01_file_io.c
    [ ] 02_preprocessor.c
    [ ] 03_linked_list.c
    [ ] 04_stack.c
    [ ] 06_header_files/ (multi-file project)

[ ] Phase 5: Real Projects (2/2 files)
    [ ] 01_contact_manager.c
    [ ] 02_text_analyzer.c

Total: [ ] 26/26 programs completed
```

---

## Timeline Estimates

### By Phase
- **Phase 1**: 10-15 hours (7 files × 1-2 hours each)
- **Phase 2**: 15-20 hours (6 files × 2-3 hours each)
- **Phase 3**: 15-20 hours (5 files × 3-4 hours each)
- **Phase 4**: 12-16 hours (4 files × 3-4 hours each)
- **Phase 5**: 10-15 hours (2 projects × 5-8 hours each)

**Total**: 62-86 hours (approximately 16 weeks at 4-6 hours/week)

### Daily Routine (1-2 hours)

1. **Read** (15 min): Open the file, read all comments
2. **Understand** (20 min): Run the program, experiment
3. **Practice** (30 min): Do 2-3 exercises
4. **Build** (30 min): Create your own variation

### Weekly Goals

- **Days 1-5**: Complete files in sequence
- **Day 6**: Review and do additional exercises
- **Day 7**: Build a small project using the week's concepts

---

## Tools and Resources

### Compilers
- **GCC** (GNU Compiler Collection) - Standard on Linux/macOS
- **Clang** - Alternative compiler with better error messages
- **MinGW** - GCC for Windows

### Debuggers
- **GDB** (GNU Debugger) - Command-line debugger
- **LLDB** - Debugger for Clang
- **VS Code/Cursor debugger** - Visual debugging interface

### Profilers and Analysis Tools
- **Valgrind** - Memory leak detection (Linux/macOS)
- **gprof** - Performance profiling
- **Address Sanitizer** - Memory error detection (`-fsanitize=address`)

### IDEs and Editors
- **VS Code / Cursor** - Recommended, with C/C++ extension
- **CLion** - JetBrains IDE (commercial)
- **Vim / Emacs** - Classic text editors
- **Sublime Text** - Lightweight editor

### Online Resources

**Books**:
- "The C Programming Language" by Kernighan & Ritchie (K&R) - The classic
- "C Programming: A Modern Approach" by K.N. King - Comprehensive
- "Expert C Programming: Deep C Secrets" by Peter van der Linden - Advanced

**Websites**:
- [Learn-C.org](https://www.learn-c.org/) - Interactive tutorials
- [CS50](https://cs50.harvard.edu/) - Harvard's free course
- [C Reference](https://en.cppreference.com/w/c) - Language reference

**Practice Platforms**:
- [LeetCode](https://leetcode.com/) - Start with Easy problems
- [HackerRank](https://www.hackerrank.com/domains/c) - C challenges
- [Exercism](https://exercism.org/tracks/c) - Mentored practice

---

## Tips for Success

### Best Practices

**DO**:
- Read every comment carefully
- Type code yourself (don't copy-paste)
- Do ALL exercises
- Use the debugger frequently
- Experiment and break things
- Take notes on key concepts
- Review regularly

**DON'T**:
- Skip ahead to "cool stuff"
- Just read without coding
- Copy code without understanding
- Ignore compiler warnings
- Rush through material
- Give up when stuck

### Understanding Compiler Errors

**Common errors and solutions**:

```c
// Error: 'x' undeclared
int result = x + 5;  // ❌

int x = 10;          // ✅ Declare before using
int result = x + 5;
```

```c
// Error: expected ';' before '}'
int x = 5
}  // ❌

int x = 5;  // ✅ Add semicolon
}
```

```c
// Warning: format '%d' expects 'int', but argument has type 'double'
double d = 3.14;
printf("%d", d);  // ❌

printf("%f", d);  // ✅ Use correct format specifier
```

### Common Mistakes to Avoid

```c
// ❌ BAD: Uninitialized variable
int x;
printf("%d", x);  // Undefined behavior!

// ✅ GOOD: Initialize variables
int x = 0;
printf("%d", x);
```

```c
// ❌ BAD: Buffer overflow
int arr[5];
arr[10] = 42;  // Accessing out of bounds!

// ✅ GOOD: Stay within bounds
int arr[5];
arr[4] = 42;  // Last valid index is size-1
```

```c
// ❌ BAD: Memory leak
int *ptr = malloc(100 * sizeof(int));
// ... use ptr ...
// Forgot to free!

// ✅ GOOD: Always free allocated memory
int *ptr = malloc(100 * sizeof(int));
// ... use ptr ...
free(ptr);
ptr = NULL;  // Good practice
```

```c
// ❌ BAD: Returning pointer to local variable
int* getNumber() {
    int x = 42;
    return &x;  // x is destroyed after function returns!
}

// ✅ GOOD: Use dynamic allocation or static
int* getNumber() {
    static int x = 42;
    return &x;
}
```

### Debugging Strategies

1. **Read error messages carefully** - They tell you exactly what's wrong
2. **Use printf debugging** - Print variable values at key points
3. **Use GDB or IDE debugger** - Step through code line by line
4. **Rubber duck debugging** - Explain code to someone (or something)
5. **Simplify** - Comment out code to isolate the problem
6. **Check assumptions** - Verify array sizes, null pointers, etc.

---

## Key Concepts

### Memory Management
- **Stack**: Automatic allocation, limited size, fast access
- **Heap**: Dynamic allocation (malloc/free), programmer managed
- **Memory leaks**: Failing to free allocated memory
- **Dangling pointers**: Pointers to freed memory

### Pointers
- **Definition**: Variable that stores a memory address
- **Operators**: `&` (address-of), `*` (dereference)
- **NULL**: Special value indicating "no address"
- **Arithmetic**: Adding/subtracting from pointers to navigate arrays

### Data Types
- **Primitive types**: int, float, double, char
- **Arrays**: Fixed-size collections
- **Structures**: Custom compound types
- **Type sizes**: Use `sizeof()` to determine

### Compilation Process
1. **Preprocessing**: Handle #include, #define, etc.
2. **Compilation**: Convert C code to assembly/object code
3. **Linking**: Combine object files and libraries
4. **Execution**: Run the final executable

---

## Milestone Projects

Build these after completing each phase to reinforce learning:

**After Phase 1**:
- Command-line calculator with multiple operations
- Number guessing game with lives system
- Pattern generator (stars, pyramids, diamonds)

**After Phase 2**:
- Student grade management system with arrays
- Simple text-based game (tic-tac-toe, hangman)
- Array sorting and searching with multiple algorithms

**After Phase 3**:
- Dynamic phonebook using structures and pointers
- Mini database with custom data types
- Memory allocation simulator

**After Phase 4**:
- File encryptor/decryptor with multiple algorithms
- Expression evaluator (infix to postfix)
- Custom data structure library (list, stack, queue)

**After Phase 5**:
- Your own unique project combining all concepts
- Portfolio piece demonstrating your skills

---

## Getting Help

### When You're Stuck

1. **Read the error message** - Compiler errors are very specific
2. **Check the comments** - Each file has detailed explanations
3. **Use the debugger** - Step through to see execution flow
4. **Simplify** - Remove code until you isolate the issue
5. **Review earlier concepts** - Make sure fundamentals are solid
6. **Experiment** - Try different approaches and learn from failures

### Understanding Valgrind Output

```bash
# Example valgrind output:
==1234== 40 bytes in 1 blocks are definitely lost
```

This means you allocated 40 bytes but never freed them. Find the malloc/calloc without a corresponding free.

---

## Graduation Criteria

You've completed the curriculum when you:
- Finished all 26 files
- Completed all exercises in each file
- Built both Phase 5 projects
- Created your own unique project
- Can explain pointers and memory management clearly
- Can debug programs using GDB or IDE debugger
- Can organize code across multiple files

**Congratulations! You're now an intermediate C programmer!**

---

## What's Next?

After completing this curriculum:

1. **Deepen Systems Knowledge**:
   - Learn about operating systems concepts
   - Study memory hierarchies and caching
   - Understand CPU architecture

2. **Advanced Data Structures & Algorithms**:
   - Trees (BST, AVL, B-trees)
   - Graphs and graph algorithms
   - Hash tables and advanced structures

3. **Systems Programming**:
   - POSIX system calls
   - Process management
   - Inter-process communication
   - Network programming (sockets)

4. **High-Performance Computing**:
   - Parallel programming with OpenMP
   - GPU programming with CUDA
   - Performance profiling and optimization
   - SIMD and vectorization

5. **Related Languages**:
   - **C++**: Object-oriented systems programming
   - **Rust**: Modern systems programming with safety
   - **Go**: Concurrent systems programming

6. **Contribute to Open Source**:
   - Find C projects on GitHub
   - Fix bugs, add features
   - Learn from experienced developers

---

## What Makes This Curriculum Special

1. **Comprehensive**: Covers basics through advanced topics systematically
2. **Structured**: Logical progression with clear learning phases
3. **Practical**: Real, runnable code examples you can experiment with
4. **Documented**: Extensive comments and explanations in every file
5. **Interactive**: Exercises and projects throughout to reinforce learning
6. **IDE-Ready**: Pre-configured for modern development environments
7. **Self-Paced**: Learn at your own speed, no pressure
8. **Complete**: Everything you need in one place
9. **HPC-Focused**: Prepares you for high-performance computing
10. **Battle-Tested**: Based on proven teaching methods and best practices

---

## Ready to Begin?

Start your C programming journey today:

1. Verify GCC is installed: `gcc --version`
2. Navigate to the samples directory
3. Open `phase1-foundations/01_hello_world.c`
4. Read through the file completely
5. Compile and run: `gcc -std=c11 -Wall 01_hello_world.c -o hello && ./hello`
6. Try the exercises at the end
7. Move to the next file!

**Remember**: Every expert was once a beginner. Take it one step at a time, don't be afraid to make mistakes - that's how you learn! The key is consistent practice and genuine curiosity.

Happy coding!
