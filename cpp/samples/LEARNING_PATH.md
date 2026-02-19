# C++ Learning Path for GPU Programming

## Overview
This repository contains a comprehensive C++ curriculum covering fundamentals through GPU programming preparation. Each module builds on previous concepts with hands-on examples.

## Complete Curriculum (14 Modules)

### **Module 1: C++ Fundamentals** (01-basics/)
- Hello world and compilation
- Basic I/O (cout, cin)
- Variables and data types
- Control flow (if/else, switch, loops)
- String basics (std::string)
- Functions and scope
- Enumerations (enum, enum class)
- Namespaces
- Const and constexpr
- **Why it matters**: Foundation for everything else

### **Module 2: Functions and Program Structure** (02-functions-structure/)
- Function parameters (value, reference, pointer)
- Function overloading and default arguments
- Inline functions
- Header/implementation separation (.h/.cpp)
- Preprocessor (#include, #define, #ifdef, header guards)
- Initialization forms (direct, copy, list, aggregate)
- Basic file I/O
- **Why it matters**: Proper code organization is critical in large projects; GPU code requires understanding header files

### **Module 3: Pointers and Memory** (03-pointers-memory/)
- Pointers and references
- Pointer arithmetic
- Stack vs heap memory
- Dynamic allocation (new/delete)
- Memory leaks and debugging
- nullptr vs NULL
- Arrays and pointers
- Memory alignment and padding
- **Why it matters**: GPU programming requires explicit memory management and data transfer between CPU/GPU

### **Module 4: Classes and OOP Fundamentals** (04-classes-oop/)
- Classes vs structs
- Constructors and destructors
- Member functions
- Access specifiers
- this pointer
- Copy constructor and copy assignment
- Rule of Three
- Const methods and const correctness
- Static members
- RAII pattern
- **Why it matters**: Understanding object lifetime is crucial; structs are heavily used in GPU code

### **Module 5: Inheritance and Polymorphism** (05-inheritance-polymorphism/)
- Inheritance (single and multiple)
- Access control in inheritance
- Function overriding
- Virtual functions and vtables
- Pure virtual functions and abstract classes
- Interfaces
- Virtual destructors
- Diamond problem and virtual inheritance
- Object slicing
- Type casting (static_cast, dynamic_cast, const_cast, reinterpret_cast)
- RTTI (typeid)
- **Why it matters**: Complete OOP understanding; knowing why GPU code avoids virtual functions

### **Module 6: Operator Overloading and Advanced Features** (06-operators-advanced/)
- Operator overloading (arithmetic, comparison, stream, subscript, etc.)
- Friend functions and classes
- Conversion operators
- Function call operator (functors)
- Nested classes
- Unions and type punning
- Bit manipulation
- **Why it matters**: Operator overloading is essential for vector/matrix math in graphics and GPU code

### **Module 7: Templates and Generic Programming** (07-templates/)
- Function templates
- Class templates
- Template specialization
- Variadic templates
- Non-type template parameters
- Template template parameters
- SFINAE basics
- decltype and decltype(auto)
- Type traits basics
- C++20 Concepts (brief intro)
- **Why it matters**: CUDA kernels can be templated; modern GPU libraries use templates extensively

### **Module 8: STL and Standard Library** (08-stl/)
- Containers: vector, array, list, deque
- Associative: map, set, multimap, multiset
- Unordered: unordered_map, unordered_set
- Container adapters: stack, queue, priority_queue
- Utility types: pair, tuple
- Iterators (types and usage)
- STL algorithms (sort, find, transform, etc.)
- String streams
- File streams (detailed)
- **Why it matters**: STL is everywhere in C++; understanding containers helps with CPU-side data preparation

### **Module 9: Modern C++ (C++11/14/17/20)** (09-modern-cpp/)
- Smart pointers (unique_ptr, shared_ptr, weak_ptr)
- Move semantics and rvalue references
- Rule of Five and Rule of Zero
- Lambda functions (capture, mutable, generic)
- Auto and type inference
- Range-based for loops
- Structured bindings (C++17)
- std::optional, std::variant, std::any (C++17)
- constexpr and consteval
- If-init statements (C++17)
- **Why it matters**: Modern CUDA uses C++14/17 features; lambdas work in thrust and CUDA

### **Module 10: Exception Handling and Error Management** (10-exceptions/)
- try/catch/throw
- Exception classes (std::exception hierarchy)
- Custom exceptions
- Stack unwinding
- Exception safety guarantees
- noexcept specifier
- RAII and exception safety
- Exceptions vs error codes
- assert and static_assert
- **Why it matters**: Understanding error handling (GPU code typically uses error codes, not exceptions)

### **Module 11: Multithreading and Concurrency** (11-multithreading/)
- std::thread basics
- Mutexes (mutex, recursive_mutex, timed_mutex)
- Lock guards (lock_guard, unique_lock, scoped_lock)
- Condition variables
- std::future and std::promise
- std::async
- Atomics (std::atomic)
- Memory ordering
- Race conditions and deadlocks
- Thread-local storage
- Data parallelism concepts
- **Why it matters**: Understanding parallel execution before GPU; CPU multithreading complements GPU computing

### **Module 12: Build Systems, Debugging & Tools** (12-build-debug/)
- Makefiles (basic to intermediate)
- CMake (modern practices)
- Debugging (lldb/gdb)
- Memory debugging (valgrind, AddressSanitizer)
- Profiling (Instruments, gprof)
- Unit testing (googletest intro)
- Compilation process in depth
- Static vs dynamic libraries
- **Why it matters**: Real projects need build systems; CUDA projects use CMake; debugging GPU code requires tools

### **Module 13: Advanced Topics for GPU** (13-gpu-advanced/)
- AoS vs SoA (Array of Structs vs Struct of Arrays)
- Memory access patterns and cache efficiency
- Vectorization (SIMD, compiler hints)
- Alignment and packed structures
- Function pointers and callbacks
- std::function and performance
- Math library functions
- Vector math classes (vec3, mat4)
- **Why it matters**: These patterns directly impact GPU performance; understanding memory layout is critical

### **Module 14: GPU Programming Preparation** (14-gpu-prep/)
- CPU vs GPU architecture
- Understanding memory hierarchies (registers, L1/L2, global memory)
- Bandwidth vs compute bound
- When to use GPU vs CPU
- Introduction to CUDA
- First CUDA program
- Memory transfer patterns (host to device)
- Kernel launch basics
- **Why it matters**: Bridge from C++ to GPU programming; first hands-on CUDA experience

## Learning Approach

For each module:
1. **Read** the example code and comments carefully
2. **Predict** what the code will do before compiling
3. **Compile and run** using VS Code (`Cmd+Shift+B`) or command line
4. **Experiment** with the "TRY THIS" suggestions
5. **Debug** if something doesn't work as expected
6. **Move forward** when you understand the concepts

## Quick Reference

### Compilation Commands
```bash
# Basic compilation
g++ -std=c++17 -o program program.cpp

# With debugging symbols
g++ -std=c++17 -g -o program program.cpp

# With warnings (recommended)
g++ -std=c++17 -Wall -Wextra -o program program.cpp

# With optimization
g++ -std=c++17 -O2 -o program program.cpp
```

### VS Code Usage
- **Build and Run**: `Cmd+Shift+B` → select "Build and Run C++ (current file)"
- **Debug**: `F5` (set breakpoints by clicking left of line numbers)
- **See .vscode/README.md** for detailed instructions

## Module Completion Checklist

Track your progress:
- [ ] Module 1: C++ Fundamentals
- [ ] Module 2: Functions and Program Structure
- [ ] Module 3: Pointers and Memory
- [ ] Module 4: Classes and OOP Fundamentals
- [ ] Module 5: Inheritance and Polymorphism
- [ ] Module 6: Operator Overloading
- [ ] Module 7: Templates
- [ ] Module 8: STL and Standard Library
- [ ] Module 9: Modern C++
- [ ] Module 10: Exception Handling
- [ ] Module 11: Multithreading
- [ ] Module 12: Build Systems & Tools
- [ ] Module 13: Advanced Topics for GPU
- [ ] Module 14: GPU Programming Preparation

## Estimated Timeline

- **Modules 1-2**: 1-2 weeks (fundamentals)
- **Modules 3-6**: 2-3 weeks (OOP and memory)
- **Modules 7-9**: 2-3 weeks (templates and modern C++)
- **Modules 10-11**: 1-2 weeks (exceptions and threading)
- **Modules 12-14**: 1-2 weeks (tools and GPU prep)

**Total**: 7-12 weeks for comprehensive coverage (adjust based on your pace)

## Next Steps

1. Start with Module 1 in the `01-basics/` directory
2. Work through examples sequentially
3. Don't rush - understanding is more important than speed
4. When ready for GPU programming, Module 14 will bridge you to CUDA
5. After Module 14, you'll be ready for dedicated GPU programming courses

## Resources

- **C++ Reference**: https://cppreference.com
- **CUDA Documentation**: https://docs.nvidia.com/cuda/
- **Modern C++ Features**: https://github.com/AnthonyCalandra/modern-cpp-features

Good luck on your C++ and GPU programming journey!
