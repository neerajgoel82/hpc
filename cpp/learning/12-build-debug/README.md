# Module 12: Build Systems, Debugging & Tools

## Overview
Learn essential tools for real-world C++ development: build systems, debuggers, profilers, and testing frameworks. Critical for managing GPU projects.

## Topics Covered

### Makefiles
- Makefile basics
- Rules and targets
- Variables in Makefiles
- Pattern rules
- Dependencies
- Phony targets
- Automatic variables ($@, $<, $^)
- Common patterns

### CMake
- **Modern build system** (industry standard)
- CMakeLists.txt basics
- **add_executable**, **add_library**
- **target_link_libraries**
- **find_package**
- CMake variables
- Generator expressions
- Out-of-source builds
- Cross-platform builds
- CMake for CUDA projects

### Compilation Process
- Preprocessing
- Compilation
- Assembly
- Linking
- Object files (.o)
- Static libraries (.a)
- Dynamic/shared libraries (.so, .dylib, .dll)
- Symbol tables

### Compiler Flags
- Optimization levels (-O0, -O1, -O2, -O3, -Ofast)
- Debug symbols (-g)
- Warnings (-Wall, -Wextra, -Werror)
- Standards (-std=c++17, -std=c++20)
- Include paths (-I)
- Library paths (-L)
- Linking libraries (-l)

### Debugging with lldb/gdb
- Starting debugger
- Breakpoints
- Stepping (step, next, finish)
- Examining variables
- Call stack
- Watchpoints
- Conditional breakpoints
- Debugging multi-threaded programs

### Memory Debugging
- **Valgrind** - Memory leak detection
- **AddressSanitizer** - Memory error detection
- **LeakSanitizer** - Leak detection
- **ThreadSanitizer** - Race condition detection
- **UndefinedBehaviorSanitizer** - UB detection

### Profiling
- **gprof** - CPU profiling
- **Instruments** (macOS) - System profiling
- **perf** (Linux) - Performance analysis
- Flame graphs
- Hotspot identification

### Static Analysis
- Compiler warnings
- **clang-tidy** - Linter
- **cppcheck** - Static analyzer
- **Clang Static Analyzer**

### Unit Testing
- **Google Test (gtest)** basics
- Test fixtures
- Assertions (EXPECT, ASSERT)
- Test organization
- Running tests
- Test-driven development (TDD)

### Version Control Integration
- Git hooks for testing
- CI/CD basics
- Automated builds
- Code coverage

## Why This Matters for GPU

### CUDA Build Systems
- CUDA projects use CMake
- nvcc compiler integration
- Separable compilation for CUDA
- Host and device code compilation
- Managing CUDA architectures

### Example CMakeLists.txt for CUDA
```cmake
cmake_minimum_required(VERSION 3.18)
project(MyGPUProject CUDA CXX)

find_package(CUDA REQUIRED)

add_executable(my_kernel kernel.cu main.cpp)
set_target_properties(my_kernel PROPERTIES
    CUDA_ARCHITECTURES "75;80;86")
```

### Debugging GPU Code
- **cuda-gdb** - GPU debugger
- **Nsight** - NVIDIA debugging/profiling
- **compute-sanitizer** - GPU memory checker
- Detecting GPU memory leaks
- Finding race conditions in kernels

### Profiling GPU Code
- **nvprof** - CUDA profiler (deprecated)
- **Nsight Systems** - System-wide profiling
- **Nsight Compute** - Kernel profiling
- Identifying GPU bottlenecks
- Memory bandwidth analysis

### Real CUDA Project Structure
```
project/
├── CMakeLists.txt
├── include/
│   └── kernels.cuh
├── src/
│   ├── main.cpp
│   └── kernels.cu
├── tests/
│   └── test_kernels.cpp
└── build/
```

## Coming Soon

Practical examples including:
- Makefile for multi-file projects
- CMake for C++ and CUDA projects
- Debugging session walkthroughs
- Valgrind usage examples
- Profiling and optimization workflow
- Unit testing with Google Test
- Complete GPU project build setup

## Estimated Time
15-20 hours

## Prerequisites
Complete Modules 1-11 first.

**Note**: These tools are essential for professional development!