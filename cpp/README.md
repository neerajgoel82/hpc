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

# Using test script (if available)
./test_all_modules.sh
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

- Module-specific README files (when migrated)
- Check `LEARNING_PATH.md` for detailed curriculum
- See `COMPILATION_STATUS.md` for sample verification

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

## Tips

- Always compile with `-Wall -Wextra`
- Use `valgrind` to check memory issues
- Leverage the STL - don't reinvent the wheel
- Read compiler error messages (especially template errors)
- Understand move semantics for performance

## Performance Considerations

For HPC applications:
- Consider cache locality
- Use move semantics for large objects
- Reserve space in vectors when size is known
- Profile before optimizing
- Understand RVO (Return Value Optimization)

## Next Steps

- Apply concepts in [projects/](projects/)
- Explore [CUDA](../cuda/) for GPU acceleration
- Compare with [C](../c/) implementations
- Use [Python](../python/) for prototyping

---

**Note**: This directory structure is ready for content migration from existing cpp-samples repository.
