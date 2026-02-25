# Module 7: Templates and Generic Programming - Homework

This directory contains exercise files for practicing C++ concepts learned in classwork.

## What's in Homework?

Homework files are **practice exercises** that:
- List "TRY THIS" exercises from classwork
- Provide a template structure
- Give you space to implement solutions
- Test your understanding of C++ concepts

## How to Use

### 1. Read Classwork First
Always read and understand the corresponding classwork file before attempting homework.

### 2. Read the Exercises
Each homework file contains exercises extracted from the classwork file.

### 3. Implement Solutions
Write your C++ code in the homework file:
```cpp
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
- Read compiler error messages carefully
- Use std::cout for debugging
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

1. **Study**: Read ../classwork/01_hello_world.cpp
2. **Practice**: Edit homework/01_hello_world.cpp
3. **Compile**: `make 01_hello_world`
4. **Test**: `./01_hello_world`
5. **Iterate**: Debug and improve

## Tips for Success

- **Start Simple**: Begin with the first exercise
- **Test Often**: Compile after each change
- **Use Comments**: Document your approach
- **Handle Errors**: Use try-catch where appropriate
- **Modern C++**: Use C++17 features when suitable
- **Be Patient**: C++ can be challenging but rewarding

## Common Issues

### Compilation Errors
```bash
# Read error messages carefully
g++ -std=c++17 -Wall -Wextra 01_hello_world.cpp -o 01_hello_world

# Common fixes:
# - Missing semicolon
# - Undeclared variable
# - Type mismatch
# - Missing header file (#include)
# - Namespace issues (std::)
```

### Runtime Errors
- Use std::cout to debug
- Check array bounds
- Verify pointer validity
- Handle exceptions appropriately
- Test with simple inputs first

### Linking Errors
- Make sure all necessary files are compiled
- Check for missing library flags (e.g., -pthread)
- Verify function implementations

## C++ Best Practices

- Use `const` where appropriate
- Prefer references over pointers when possible
- Use smart pointers for dynamic memory
- Initialize variables at declaration
- Use meaningful variable names
- Follow RAII principles

## Completion Checklist

Mark exercises as complete when:
- [ ] Code compiles without warnings
- [ ] Program runs correctly
- [ ] Edge cases handled
- [ ] Code follows modern C++ practices
- [ ] You understand why it works

## Next Steps

After completing homework:
1. Compare with classwork approach
2. Optimize if needed
3. Explore modern C++ alternatives
4. Move to next file
5. Review periodically
