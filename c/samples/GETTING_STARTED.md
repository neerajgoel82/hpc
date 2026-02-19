# Getting Started with C Programming

This guide will help you set up your environment and start learning C programming using this curriculum.

## Prerequisites

Before you begin, make sure you have:
1. **Cursor IDE** (or VS Code) - You already have this!
2. **GCC Compiler** - For compiling C programs
3. **Basic terminal/command line knowledge**

## Verifying Your Setup

Open a terminal in Cursor and run:

```bash
gcc --version
```

You should see the GCC version information. If not, you need to install GCC:

### Installing GCC

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
- Install MinGW or use WSL (Windows Subsystem for Linux)

## Quick Start

### Method 1: Using Cursor's Build Tasks (Recommended)

1. Open any `.c` file (start with `phase1-foundations/01_hello_world.c`)
2. Press `Cmd+Shift+B` (Mac) or `Ctrl+Shift+B` (Windows/Linux)
3. Select "Build and Run" from the menu
4. The program will compile and run automatically!

### Method 2: Using the Makefile

From the terminal in the project root:

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

### Method 3: Manual Compilation

```bash
# Compile
gcc -Wall -Wextra -g phase1-foundations/01_hello_world.c -o hello

# Run
./hello
```

## Learning Path

Follow this structured path:

### Week 1-2: Phase 1 - Foundations
Start here if you're new to C:
1. `01_hello_world.c` - Your first program
2. `02_data_types.c` - Understanding variables
3. `03_operators.c` - Performing operations
4. `04_input_output.c` - User interaction
5. `05_if_else.c` - Making decisions
6. `06_switch.c` - Multiple choices
7. `07_loops.c` - Repetition

**Daily Practice:** Spend 30-60 minutes, complete exercises at the end of each file.

### Week 3-5: Phase 2 - Building Blocks
1. `01_functions.c` - Writing reusable code
2. `02_function_scope.c` - Variable lifetime
3. `03_recursion.c` - Functions calling themselves
4. `04_arrays.c` - Collections of data
5. `05_multidimensional_arrays.c` - Matrices
6. `06_strings.c` - Text processing

**Practice Projects:**
- Simple calculator
- Number guessing game
- Grade calculator

### Week 6-9: Phase 3 - Core Concepts
1. `01_pointers_basics.c` - Understanding memory
2. `02_pointers_arrays.c` - Pointers and arrays
3. `03_pointers_functions.c` - Advanced pointer usage
4. `04_structures.c` - Custom data types
5. `05_dynamic_memory.c` - Runtime memory allocation

**Practice Projects:**
- Student record system
- Simple database

### Week 10-14: Phase 4 - Advanced Topics
1. `01_file_io.c` - Reading and writing files
2. `02_preprocessor.c` - Compile-time features
3. `03_linked_list.c` - Dynamic data structures
4. `04_stack.c` - LIFO data structure

**Practice Projects:**
- File encryption program
- Expression evaluator

### Week 15-16: Phase 5 - Real Projects
1. `01_contact_manager.c` - Full CRUD application
2. `02_text_analyzer.c` - File processing

**Challenge:** Build your own project combining all concepts!

## Tips for Success

### 1. Read, Understand, Modify, Create

For each file:
- **Read** the code and comments carefully
- **Understand** what each part does
- **Modify** the code and see what happens
- **Create** your own variations

### 2. Do the Exercises

Every file has exercises at the bottom. These are crucial for learning!

### 3. Debug Like a Pro

Use the debugger (press `F5` in Cursor):
- Set breakpoints by clicking left of line numbers
- Step through code line by line
- Inspect variable values
- Understand program flow

### 4. Common Mistakes to Avoid

```c
// ❌ BAD: Uninitialized variable
int x;
printf("%d", x);  // Undefined behavior!

// ✅ GOOD: Initialize variables
int x = 0;
printf("%d", x);

// ❌ BAD: Accessing out of bounds
int arr[5];
arr[10] = 42;  // Buffer overflow!

// ✅ GOOD: Stay within bounds
int arr[5];
arr[4] = 42;  // Last valid index is size-1

// ❌ BAD: Memory leak
int *ptr = malloc(100);
// ... use ptr ...
// Forgot to free!

// ✅ GOOD: Always free allocated memory
int *ptr = malloc(100);
// ... use ptr ...
free(ptr);
```

### 5. Keyboard Shortcuts in Cursor

- `Cmd+Shift+B` - Build/Run program
- `F5` - Start debugging
- `F9` - Toggle breakpoint
- `F10` - Step over
- `F11` - Step into
- `Shift+F11` - Step out

## Debugging Your First Program

Let's debug `01_hello_world.c`:

1. Open `phase1-foundations/01_hello_world.c`
2. Click left of line `int main()` to set a breakpoint (red dot appears)
3. Press `F5` to start debugging
4. Use `F10` to step through each line
5. Watch how the program executes line by line

## Getting Help

### When You're Stuck

1. **Read the error message** - Compiler errors tell you what's wrong
2. **Check the comments** - Each file has detailed explanations
3. **Use the debugger** - Step through to see what's happening
4. **Experiment** - Try different things and learn from mistakes

### Understanding Compiler Errors

```bash
# Example error:
error: 'x' undeclared (first use in this function)
```

This means you used variable `x` without declaring it. Solution:
```c
int x;  // Declare before using
```

## Next Steps

1. **Start with Phase 1** - Don't skip ahead!
2. **Code along** - Type the code yourself, don't just read
3. **Do exercises** - They reinforce learning
4. **Build projects** - Apply what you've learned
5. **Review regularly** - Revisit earlier concepts

## Recommended Resources

### Books
- "The C Programming Language" by Kernighan & Ritchie (K&R)
- "C Programming: A Modern Approach" by K.N. King

### Online
- [Learn-C.org](https://www.learn-c.org/) - Interactive tutorials
- [CS50](https://cs50.harvard.edu/) - Harvard's free course

### Practice Platforms
- [LeetCode](https://leetcode.com/) - Start with Easy problems
- [HackerRank](https://www.hackerrank.com/domains/c) - C challenges

## Study Schedule Suggestion

**Beginner (Complete in 16 weeks):**
- **Week 1-2:** Phase 1 (1-2 files per day)
- **Week 3-5:** Phase 2 (1 file per day)
- **Week 6-9:** Phase 3 (slower pace, harder concepts)
- **Week 10-14:** Phase 4 (1 file per day)
- **Week 15-16:** Phase 5 (1 week per project)

**Intensive (Complete in 8 weeks):**
- **Week 1:** Phase 1 (all files)
- **Week 2-3:** Phase 2
- **Week 4-5:** Phase 3
- **Week 6-7:** Phase 4
- **Week 8:** Phase 5

**Self-Paced:**
- Go at your own speed
- Focus on understanding, not speed
- Quality over quantity!

## Your First Day

Here's what to do right now:

1. Open `phase1-foundations/01_hello_world.c`
2. Read through the entire file
3. Press `Cmd+Shift+B` and select "Build and Run"
4. See your first program run!
5. Modify the message and run again
6. Try the exercises at the bottom
7. Move to `02_data_types.c`

## Tracking Your Progress

Use the checklist in README.md to track completion:
- [ ] Phase 1: Foundations
- [ ] Phase 2: Building Blocks
- [ ] Phase 3: Core Concepts
- [ ] Phase 4: Advanced Topics
- [ ] Phase 5: Projects

---

**Ready to begin?** Open `phase1-foundations/01_hello_world.c` and start your C programming journey!

**Remember:** Every expert was once a beginner. Take it one step at a time, and don't be afraid to make mistakes - that's how you learn!

Happy coding! 🚀
