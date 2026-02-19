# C++ Programming - Claude Instructions

## Curriculum Structure

The C++ samples are organized by modules:
- **01-basics**: Syntax, I/O, variables, control flow
- **02-functions-structure**: Functions, references, program structure
- **03-pointers-memory**: Pointers, memory management, smart pointers
- **04-classes-oop**: Classes, objects, constructors, destructors
- **05-inheritance-polymorphism**: OOP principles, virtual functions
- **06-operators-advanced**: Operator overloading, advanced features
- **07-templates**: Template functions and classes
- **08-stl**: Standard Template Library containers and algorithms
- **09-modern-cpp**: C++11/14/17 features
- **10-exceptions**: Exception handling
- **11-multithreading**: Threading, synchronization
- **12-build-debug**: Build systems, debugging
- **13-gpu-advanced**: Advanced GPU concepts
- **14-gpu-prep**: GPU preparation

## Compilation Standards

### Required Flags
```bash
g++ -std=c++17 -Wall -Wextra -O2 file.cpp -o output
```

- Use C++17 standard (`-std=c++17`)
- Enable all warnings (`-Wall -Wextra`)
- No warnings tolerated
- Use `-O2` for performance samples

### Build System
- Test scripts exist (e.g., `test_all_modules.sh`)
- May use Makefiles or CMake
- Follow existing build patterns

## Coding Style

### Naming Conventions
- Classes: `PascalCase` (e.g., `MyClass`, `VectorArray`)
- Functions/Methods: `camelCase` or `snake_case` (be consistent)
- Variables: `camelCase` or `snake_case`
- Constants: `kConstantName` or `CONSTANT_NAME`
- Member variables: `m_name` or `name_` (trailing underscore)

### Modern C++ Preferences
- Use `auto` when type is obvious
- Use range-based for loops when possible
- Prefer smart pointers over raw pointers
- Use `nullptr` not `NULL`
- Use `std::array` or `std::vector` over C arrays
- Prefer `using` over `typedef`

### Code Organization
```cpp
// Order:
1. Includes (<> first, then "")
2. Using declarations (if any, minimal)
3. Class declarations
4. Main function (for samples)
5. Function/method implementations
```

## Module-Specific Guidelines

### Modules 01-03: Basics and Fundamentals
- Can use C-style when teaching basics
- Introduce C++ features gradually
- Keep samples focused on one concept

### Modules 04-06: OOP Fundamentals
- Demonstrate proper class design
- Rule of Three/Five when appropriate
- Show const correctness

### Modules 07-09: Advanced C++
- Use modern C++ features
- Template best practices
- STL idioms and patterns

### Modules 10-14: Advanced Topics
- Production-quality code
- Full error handling
- Thread safety when relevant
- Performance considerations

## Common Patterns

### Resource Management
```cpp
// Prefer RAII
std::vector<int> vec;  // automatic cleanup

// Smart pointers when needed
std::unique_ptr<T> ptr = std::make_unique<T>();
std::shared_ptr<T> ptr = std::make_shared<T>();

// Avoid raw new/delete in modern C++
```

### Error Handling
```cpp
// Early modules: simple
if (!success) return 1;

// Later modules: exceptions
try {
    operation();
} catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
}
```

### Input/Output
```cpp
// Use iostream
std::cout << "Value: " << value << '\n';  // prefer '\n' over std::endl

// For performance: std::ios_base::sync_with_stdio(false);
```

## What NOT to Do

- Don't use C-style casts in modern C++ code (use `static_cast`, etc.)
- Don't use `new`/`delete` when containers or smart pointers work
- Don't use raw pointers for ownership
- Don't ignore the Rule of Three/Five when needed
- Don't catch exceptions by value (use const reference)
- Don't use `using namespace std;` in headers
- Don't add `std::` to every line if teaching basic concepts (be pragmatic)

## When Adding New Samples
1. Place in appropriate module directory
2. Match the module's complexity level
3. Follow existing naming patterns
4. Compile and test
5. Update module documentation if needed

## Testing
- Compile with all warnings: `g++ -std=c++17 -Wall -Wextra -Werror`
- Run and verify output
- Use existing test scripts
- For memory-intensive samples: valgrind

## Build Systems
- Modules may use Makefiles or CMake
- Follow the existing pattern in each module
- Don't mix build systems unnecessarily

## Performance Considerations
- For HPC code, consider:
  - Cache locality (data layout)
  - Move semantics for large objects
  - Reserve space in vectors when size is known
  - Const correctness for optimization opportunities
  - RVO/NRVO (return value optimization)

## Documentation
- Add comments for complex algorithms
- Document class interfaces
- Don't over-comment obvious code
- Use clear, descriptive names that reduce need for comments
