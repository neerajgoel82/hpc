# C++ Learning Path for GPU Programming

A comprehensive, hands-on curriculum covering C++ fundamentals through GPU programming with CUDA. This repository provides 14 structured modules with 46+ practical examples designed to take you from C++ basics to writing efficient GPU code.

## 🎯 What You'll Learn

- **Complete C++ fundamentals** - Syntax, types, control flow, functions
- **Object-Oriented Programming** - Classes, inheritance, polymorphism
- **Modern C++** - C++11/14/17/20 features, smart pointers, lambdas, move semantics
- **Memory management** - Pointers, references, dynamic allocation, RAII
- **Templates and STL** - Generic programming, containers, algorithms
- **Concurrency** - Multithreading, atomics, parallel patterns
- **Build systems** - Makefiles, CMake, debugging, profiling
- **GPU-specific concepts** - Memory layouts (AoS vs SoA), optimization patterns
- **CUDA programming** - Your first GPU programs

---

## 🚀 Quick Start

### Prerequisites
- macOS, Linux, or Windows with WSL
- C++ compiler with C++17 support (g++ or clang)
- VS Code or Cursor IDE (optional but recommended)

### Getting Started
```bash
# Navigate to the C++ learning directory
cd cpp/learning

# Start with Module 1
cd 01-basics

# Compile and run your first program
g++ -std=c++17 -o hello 01_hello_world.cpp
./hello
```

### Using VS Code/Cursor
1. Open any `.cpp` file
2. Press `Cmd+Shift+B` (or `Ctrl+Shift+B` on Windows/Linux)
3. Select "Build and Run C++ (current file)"
4. Press `F5` to debug with breakpoints

See `.vscode/README.md` for detailed IDE setup.

---

## 📚 Complete Curriculum (14 Modules)

### **Fundamentals (Modules 1-3)**

#### Module 1: C++ Fundamentals (9 files)
**Location**: `01-basics/`

Core topics:
- Hello world and compilation
- Basic I/O (cout, cin)
- Variables and data types
- Control flow (if/else, switch, loops)
- String basics (std::string)
- Functions and scope
- Enumerations (enum, enum class)
- Namespaces
- Const and constexpr

**Why it matters**: Foundation for everything else

**Estimated time**: 1-2 weeks

---

#### Module 2: Functions and Program Structure (7 files)
**Location**: `02-functions-structure/`

Core topics:
- Function parameters (value, reference, pointer)
- Function overloading and default arguments
- Inline functions
- Header/implementation separation (.h/.cpp)
- Preprocessor (#include, #define, #ifdef, header guards)
- Initialization forms (direct, copy, list, aggregate)
- Basic file I/O

**Why it matters**: Proper code organization is critical in large projects; GPU code requires understanding header files

**Estimated time**: 1-2 weeks

---

#### Module 3: Pointers and Memory (4 files)
**Location**: `03-pointers-memory/`

Core topics:
- Pointers and references
- Pointer arithmetic
- Stack vs heap memory
- Dynamic allocation (new/delete)
- Memory leaks and debugging
- nullptr vs NULL
- Arrays and pointers
- Memory alignment and padding

**Why it matters**: GPU programming requires explicit memory management and data transfer between CPU/GPU

**Estimated time**: 2-3 weeks

---

### **Object-Oriented Programming (Modules 4-6)**

#### Module 4: Classes and OOP Fundamentals (4 files)
**Location**: `04-classes-oop/`

Core topics:
- Classes vs structs
- Constructors and destructors
- Member functions
- Access specifiers (public, private, protected)
- this pointer
- Copy constructor and copy assignment
- Rule of Three
- Const methods and const correctness
- Static members
- RAII pattern (Resource Acquisition Is Initialization)

**Why it matters**: Understanding object lifetime is crucial; structs are heavily used in GPU code; RAII is essential for managing GPU resources

**Estimated time**: 2-3 weeks

---

#### Module 5: Inheritance and Polymorphism (4 files)
**Location**: `05-inheritance-polymorphism/`

Core topics:
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

**Why it matters**: Complete OOP understanding; knowing why GPU code avoids virtual functions

**Estimated time**: 2-3 weeks

---

#### Module 6: Operator Overloading and Advanced Features (3 files)
**Location**: `06-operators-advanced/`

Core topics:
- Operator overloading (arithmetic, comparison, stream, subscript, etc.)
- Friend functions and classes
- Conversion operators
- Function call operator (functors)
- Nested classes
- Unions and type punning
- Bit manipulation
- Vector math classes (Vec3)
- Matrix operators (4x4 matrices)

**Why it matters**: Operator overloading is essential for vector/matrix math in graphics and GPU code

**Estimated time**: 1-2 weeks

---

### **Advanced C++ (Modules 7-10)**

#### Module 7: Templates and Generic Programming (2 files)
**Location**: `07-templates/`

Core topics:
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

**Why it matters**: CUDA kernels can be templated; modern GPU libraries use templates extensively

**Estimated time**: 2-3 weeks

---

#### Module 8: STL and Standard Library (2 files)
**Location**: `08-stl/`

Core topics:
- Containers: vector, array, list, deque
- Associative: map, set, multimap, multiset
- Unordered: unordered_map, unordered_set
- Container adapters: stack, queue, priority_queue
- Utility types: pair, tuple
- Iterators (types and usage)
- STL algorithms (sort, find, transform, etc.)
- String streams
- File streams (detailed)

**Why it matters**: STL is everywhere in C++; understanding containers helps with CPU-side data preparation; Thrust library mirrors STL

**Estimated time**: 2-3 weeks

---

#### Module 9: Modern C++ (C++11/14/17/20) (3 files)
**Location**: `09-modern-cpp/`

Core topics:
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

**Why it matters**: Modern CUDA uses C++14/17 features; lambdas work in Thrust and CUDA; efficient resource transfers

**Estimated time**: 2-3 weeks

---

#### Module 10: Exception Handling and Error Management (1 file)
**Location**: `10-exceptions/`

Core topics:
- try/catch/throw
- Exception classes (std::exception hierarchy)
- Custom exceptions
- Stack unwinding
- Exception safety guarantees
- noexcept specifier
- RAII and exception safety
- Exceptions vs error codes
- assert and static_assert

**Why it matters**: Understanding error handling (GPU code typically uses error codes, not exceptions)

**Estimated time**: 1 week

---

### **Parallel Programming (Modules 11-14)**

#### Module 11: Multithreading and Concurrency (2 files)
**Location**: `11-multithreading/`

Core topics:
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

**Why it matters**: Understanding parallel execution before GPU; CPU multithreading complements GPU computing

**Estimated time**: 2-3 weeks

**Compilation note**: Requires `-pthread` flag

---

#### Module 12: Build Systems, Debugging & Tools (2 files)
**Location**: `12-build-debug/`

Core topics:
- Makefiles (basic to intermediate)
- CMake (modern practices)
- Debugging (lldb/gdb)
- Memory debugging (valgrind, AddressSanitizer)
- Profiling (Instruments, gprof)
- Unit testing (googletest intro)
- Compilation process in depth
- Static vs dynamic libraries

**Why it matters**: Real projects need build systems; CUDA projects use CMake; debugging GPU code requires tools

**Estimated time**: 1 week

---

#### Module 13: Advanced Topics for GPU (2 files)
**Location**: `13-gpu-advanced/`

Core topics:
- AoS vs SoA (Array of Structs vs Struct of Arrays)
- Memory access patterns and cache efficiency
- Vectorization (SIMD, compiler hints)
- Alignment and packed structures
- Function pointers and callbacks
- std::function and performance
- Math library functions
- Vector math classes (vec3, mat4)

**Why it matters**: These patterns directly impact GPU performance; understanding memory layout is CRITICAL (10-100x performance difference)

**Estimated time**: 1-2 weeks

---

#### Module 14: GPU Programming Preparation (3 files)
**Location**: `14-gpu-prep/`

Core topics:
- CPU vs GPU architecture
- Understanding memory hierarchies (registers, L1/L2, global memory)
- Bandwidth vs compute bound
- When to use GPU vs CPU
- Introduction to CUDA
- First CUDA program (simulated)
- Memory transfer patterns (host to device)
- Kernel launch basics
- Parallel patterns (map, reduce, scan, stencil)
- Memory optimization strategies

**Why it matters**: Bridge from C++ to GPU programming; first hands-on CUDA experience

**Estimated time**: 2-3 weeks

---

## 📖 How to Use This Repository

### Learning Approach

For each module:
1. **Read** example code and comments carefully
2. **Predict** what the code will do before compiling
3. **Compile and run** using VS Code (`Cmd+Shift+B`) or command line
4. **Experiment** with the "TRY THIS" suggestions in each file
5. **Debug** if something doesn't work as expected
6. **Complete** practice exercises at the end of each module
7. **Move forward** when comfortable with the concepts

### Every Example Includes:
- Detailed comments explaining concepts
- GPU relevance explanations
- "TRY THIS" exercises
- Compilation instructions
- Common mistakes to avoid
- Real-world usage examples

---

## 🔧 Compilation Guide

### Standard Compilation
```bash
# Basic compilation
g++ -std=c++17 -o program file.cpp

# With warnings (recommended)
g++ -std=c++17 -Wall -Wextra -o program file.cpp

# With debugging symbols
g++ -std=c++17 -g -o program file.cpp

# With optimization
g++ -std=c++17 -O2 -o program file.cpp
```

### Multi-file Projects
```bash
# Compile multiple source files
g++ -std=c++17 -o program file1.cpp file2.cpp

# Or compile separately and link
g++ -std=c++17 -c file1.cpp
g++ -std=c++17 -c file2.cpp
g++ -o program file1.o file2.o
```

### Multithreading Compilation
```bash
# Modules 11 and 14 require pthread
g++ -std=c++17 -pthread -o program file.cpp
```

### Complete Build with All Flags
```bash
# Recommended for development
g++ -std=c++17 -Wall -Wextra -g -O2 -o program file.cpp
```

---

## 📋 Module Completion Checklist

Track your progress through the curriculum:

**Fundamentals**
- [ ] Module 1: C++ Fundamentals ← **START HERE**
- [ ] Module 2: Functions and Program Structure
- [ ] Module 3: Pointers and Memory

**Object-Oriented Programming**
- [ ] Module 4: Classes and OOP Fundamentals
- [ ] Module 5: Inheritance and Polymorphism
- [ ] Module 6: Operator Overloading

**Advanced C++**
- [ ] Module 7: Templates
- [ ] Module 8: STL and Standard Library
- [ ] Module 9: Modern C++
- [ ] Module 10: Exception Handling

**Parallel Programming**
- [ ] Module 11: Multithreading
- [ ] Module 12: Build Systems & Tools
- [ ] Module 13: Advanced Topics for GPU
- [ ] Module 14: GPU Programming Preparation

---

## 🎓 Learning Paths

### For Complete Beginners
Start at Module 1, work sequentially through all modules.

**Timeline**: 20-24 weeks
- Modules 1-3: 4-6 weeks (fundamentals and memory)
- Modules 4-6: 4-5 weeks (OOP)
- Modules 7-10: 5-7 weeks (advanced C++)
- Modules 11-14: 5-6 weeks (parallel and GPU)

### For Programmers New to C++
Start at Module 1, but can move faster through basics. Focus on Modules 3-4 (memory/classes) and 7-9 (templates/STL/modern C++).

**Timeline**: 14-18 weeks

### For C++ Programmers Learning GPU
Review Module 3 (pointers), then focus on Modules 9-14.

**Timeline**: 8-12 weeks

---

## ⏱️ Detailed Timeline Estimates

| Phase | Modules | Topics | Estimated Time |
|-------|---------|--------|----------------|
| Fundamentals | 1-2 | Basics, functions, I/O | 1-2 weeks |
| Memory | 3 | Pointers, references, dynamic memory | 2-3 weeks |
| OOP | 4-6 | Classes, inheritance, operators | 4-5 weeks |
| Templates & Modern C++ | 7-9 | Templates, STL, C++11/14/17 | 5-7 weeks |
| Error Handling | 10 | Exceptions | 1 week |
| Parallelism | 11 | Multithreading | 2-3 weeks |
| Tools | 12 | Build systems, debugging | 1 week |
| GPU Prep | 13-14 | Memory patterns, CUDA intro | 2-4 weeks |

**Total**: 18-24 weeks for comprehensive mastery

*Adjust based on your pace and prior experience*

---

## 💡 Why This Curriculum?

### GPU-Focused Approach
Every concept is taught with GPU programming in mind:
- Data types matter for GPU performance
- Memory management is critical for CPU↔GPU transfers
- Understanding parallelism prepares you for GPU threads
- Build systems are essential for CUDA projects
- Memory layout (AoS vs SoA) can mean 10-100x performance difference

### Comprehensive Coverage
- All core C++ concepts (syntax to advanced features)
- Complete OOP knowledge
- Modern C++ features (C++11/14/17)
- Real-world tools and practices
- Direct path to GPU programming

### Hands-On Learning
- 46+ code examples
- 100+ "TRY THIS" exercises
- Practical, real-world patterns
- Performance-oriented approach
- All examples compile and run

---

## 🛠️ Tools You'll Master

- **Compilers**: g++, clang, nvcc (CUDA)
- **Build Systems**: Make, CMake
- **Debuggers**: lldb, gdb, cuda-gdb
- **Profilers**: gprof, Nsight, Instruments
- **Memory Tools**: Valgrind, AddressSanitizer
- **Testing**: Google Test
- **Version Control**: Git (for projects)

---

## 📊 Current Status

- ✅ **All 14 Modules**: Complete
- ✅ **46+ Example Files**: Created and tested
- ✅ **Compilation Status**: 100% Success
- ✅ **7,000+ Lines of Code**: With extensive comments
- ✅ **VS Code Integration**: Configured
- ✅ **Build Systems**: Makefiles and CMake examples

---

## 🎯 What You'll Master

By completing this curriculum, you will:

### C++ Fundamentals
- Variables, types, control flow
- Functions and scope
- Pointers and memory management
- Object-oriented programming
- Templates and generic programming
- STL containers and algorithms
- Modern C++ features (C++11/14/17)

### Advanced Topics
- Exception handling
- Multithreading and concurrency
- Build systems (Make, CMake)
- Memory optimization patterns

### GPU-Specific Knowledge
- Memory layout patterns (AoS vs SoA)
- Parallel programming concepts
- GPU architecture understanding
- CUDA programming preparation
- Performance optimization techniques

---

## 📚 Resources

### Documentation
- [C++ Reference](https://cppreference.com) - Comprehensive C++ documentation
- [CUDA Documentation](https://docs.nvidia.com/cuda/) - Official CUDA guides
- [Modern C++ Features](https://github.com/AnthonyCalandra/modern-cpp-features) - C++11/14/17/20 reference

### Books
- "C++ Primer" (Lippman, Lajoie, Moo)
- "Effective Modern C++" (Scott Meyers)
- "Programming Massively Parallel Processors" (Hwu, Kirk, Hajj)
- "CUDA by Example" (Sanders, Kandrot)

### Online Resources
- CUDA Samples: github.com/NVIDIA/cuda-samples
- LearnCpp.com for C++ basics
- NVIDIA Developer Blog
- Thrust Quick Start Guide

---

## 📱 Next Steps After Completion

### Ready for CUDA!
After Module 14, you're ready to:
1. Install NVIDIA CUDA Toolkit
2. Write your first CUDA kernel
3. Study CUDA Programming Guide
4. Explore Thrust library (GPU STL)
5. Learn cuBLAS, cuFFT (GPU libraries)
6. Build GPU-accelerated applications

### Recommended Advanced Resources
- NVIDIA CUDA Programming Guide
- GPU Gems books
- Professional CUDA C Programming
- Parallel programming courses
- GPU architecture papers

---

## 🤝 Contributing

Found an issue or have a suggestion? Feel free to open an issue or submit a pull request.

---

## 📄 License

This educational repository is provided as-is for learning purposes.

---

## 🎉 Ready to Begin?

Start with [Module 1: C++ Fundamentals](01-basics/) and work through the curriculum at your own pace.

**Remember**: Understanding is more important than speed. Take your time, experiment with the code, and complete the exercises.

Good luck on your journey from C++ to GPU programming! 🚀

---

**Created**: 2026-02-24
**Status**: Complete and Verified ✅
**Total Files**: 46+ compilable examples
**Compilation Rate**: 100% success
