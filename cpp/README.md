# C++ Programming

Learn modern C++ programming from basics through advanced topics including templates, STL, and GPU preparation.

## Structure

```
cpp/
├── samples/     # Learning samples organized by modules
├── projects/    # Complete C++ applications
└── notebooks/   # C++-related notebooks (if needed)
```

## Curriculum

The samples are organized into 14 progressive modules:

### Fundamentals (Modules 01-03)
- **01-basics**: Syntax, I/O, variables, control flow
- **02-functions-structure**: Functions, references, program organization
- **03-pointers-memory**: Pointers, memory management, smart pointers

### Object-Oriented Programming (Modules 04-06)
- **04-classes-oop**: Classes, objects, constructors, destructors
- **05-inheritance-polymorphism**: OOP principles, virtual functions
- **06-operators-advanced**: Operator overloading, advanced features

### Advanced C++ (Modules 07-09)
- **07-templates**: Template functions and classes
- **08-stl**: Standard Template Library (containers, algorithms)
- **09-modern-cpp**: C++11/14/17 features

### Systems & Parallel (Modules 10-14)
- **10-exceptions**: Exception handling
- **11-multithreading**: Threading, synchronization
- **12-build-debug**: Build systems, debugging tools
- **13-gpu-advanced**: Advanced GPU concepts
- **14-gpu-prep**: GPU programming preparation

## Getting Started

### Prerequisites
```bash
# Install G++ (if not already installed)
# macOS
brew install gcc

# Ubuntu/Debian
sudo apt-get install build-essential

# Check version (should support C++17)
g++ --version
```

### Compiling Samples

```bash
# Navigate to samples directory
cd samples/

# Compile a single file
g++ -std=c++17 -Wall -Wextra file.cpp -o output

# Test all modules (compiles all samples)
./test_all_modules.sh

# Compile specific module
cd 01-basics/
g++ -std=c++17 -Wall -Wextra 01_hello_world.cpp -o hello
./hello
```

### Compilation Standards

All C++ code uses:
- **Standard**: C++17 (`-std=c++17`)
- **Warnings**: `-Wall -Wextra`
- **Optimization**: `-O2` (for performance samples)

## Learning Path

1. **Start with 01-basics**
   - Work through modules sequentially
   - Each module builds on previous ones
   - Compile and test each example

2. **Master OOP concepts (04-06)**
   - Critical for modern C++
   - Practice class design
   - Understand inheritance and polymorphism

3. **Learn modern C++ (07-09)**
   - Templates are powerful
   - STL is essential
   - Modern features improve code quality

4. **Advanced topics (10-14)**
   - Multithreading for performance
   - GPU programming for HPC
   - Professional development tools

## Code Style

### Naming Conventions
- **Classes**: `PascalCase`
- **Functions**: `camelCase` or `snake_case` (be consistent)
- **Variables**: `camelCase` or `snake_case`
- **Constants**: `kConstantName` or `CONSTANT_NAME`

### Modern C++ Best Practices
- Use `auto` when type is obvious
- Prefer smart pointers over raw pointers
- Use range-based for loops
- Use `nullptr` not `NULL`
- RAII for resource management

## Resources

- **[samples/LEARNING_PATH.md](samples/LEARNING_PATH.md)** - Complete curriculum from basics to GPU prep
- **[samples/MODULES_SUMMARY.md](samples/MODULES_SUMMARY.md)** - Summary of all 14 modules
- **[samples/COMPILATION_STATUS.md](samples/COMPILATION_STATUS.md)** - Compilation verification status
- **[samples/COMPLETE_SUMMARY.md](samples/COMPLETE_SUMMARY.md)** - Comprehensive overview
- **samples/README.md** - Quick reference and getting started
- **samples/test_all_modules.sh** - Script to compile and test all modules
- Module-specific directories contain focused examples

## Common Patterns

### Memory Management
```cpp
// Prefer smart pointers
std::unique_ptr<int> ptr = std::make_unique<int>(42);
std::shared_ptr<Data> data = std::make_shared<Data>();

// Use containers
std::vector<int> vec = {1, 2, 3, 4, 5};
```

### Error Handling
```cpp
try {
    riskyOperation();
} catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
}
```

## Using the Test Script

The samples directory includes a comprehensive test script:

```bash
cd samples/

# Run test script to compile all modules
./test_all_modules.sh
```

The script will:
- Compile all C++ samples across all 14 modules
- Report compilation status for each file
- Show which samples compiled successfully
- Highlight any compilation errors

## Tips

- Always compile with `-Wall -Wextra`
- Use `valgrind` to check memory issues
- Leverage the STL - don't reinvent the wheel
- Read compiler error messages (especially template errors)
- Understand move semantics for performance
- Work through modules sequentially (01 → 14)

## Performance Considerations

For HPC applications:
- Consider cache locality
- Use move semantics for large objects
- Reserve space in vectors when size is known
- Profile before optimizing
- Understand RVO (Return Value Optimization)

## Sample Count

- **Module 01 (Basics)**: 24 samples covering fundamentals
- **Module 02 (Functions)**: 8 samples on program structure
- **Module 03 (Pointers)**: 4 samples on memory management
- **Module 04 (Classes)**: 5 samples on OOP basics
- **Module 05 (Inheritance)**: 4 samples on polymorphism
- **Module 06 (Operators)**: 3 samples on advanced features
- **Module 07 (Templates)**: 2 samples on generic programming
- **Module 08 (STL)**: 2 samples on standard library
- **Module 09 (Modern C++)**: 3 samples on C++11/14/17
- **Module 10 (Exceptions)**: 1 sample on error handling
- **Module 11 (Multithreading)**: 2 samples on concurrency
- **Module 12 (Build/Debug)**: 2 samples on tooling
- **Module 13 (GPU Advanced)**: 2 samples on GPU concepts
- **Module 14 (GPU Prep)**: 3 samples preparing for CUDA

**Total**: 46+ C++ source files across 14 modules

## Next Steps

- Work through modules sequentially (01 → 14)
- Apply concepts in [projects/](projects/)
- Explore [CUDA](../cuda/) for GPU acceleration
- Compare with [C](../c/) implementations for performance differences
- Use [Python](../python/) for rapid prototyping
