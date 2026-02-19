# Module 7: Templates and Generic Programming

## Overview
Master templates - one of C++'s most powerful features. Templates enable generic programming and are heavily used in modern GPU code.

## Topics Covered

### Function Templates
- Template syntax
- Template parameters
- Type deduction
- Explicit template instantiation
- Template argument deduction (C++17)

### Class Templates
- Template classes
- Template member functions
- Nested templates
- Default template arguments

### Template Specialization
- Full specialization
- Partial specialization
- When to specialize
- Specialization patterns

### Advanced Template Features
- Variadic templates (C++11)
- Parameter packs
- Fold expressions (C++17)
- Non-type template parameters
- Template template parameters

### Type Traits
- `std::is_same`
- `std::enable_if`
- `std::conditional`
- Type trait patterns
- Using traits for optimization

### SFINAE
- Substitution Failure Is Not An Error
- How SFINAE works
- SFINAE patterns
- Enabling/disabling template functions

### decltype and decltype(auto)
- `decltype` for type inference
- `decltype(auto)` for perfect return types
- Trailing return types
- When to use each

### Concepts (C++20)
- Introduction to concepts
- Concept syntax
- Constraining templates
- Standard concepts library

## Why This Matters for GPU

**CRUCIAL FOR GPU PROGRAMMING:**

### CUDA Template Kernels
```cpp
template<typename T>
__global__ void add(T* a, T* b, T* c, int n) {
    // Works with float, double, int, etc.
}
```

### GPU Libraries Use Templates
- **Thrust**: Heavily templated (like STL for GPU)
- **CUB**: Template-based GPU algorithms
- **Eigen**: Matrix library with templates
- **Modern CUDA**: Templates everywhere

### Compile-Time Optimization
- Templates generate specialized code for each type
- No runtime overhead
- Compiler optimizes each instantiation
- Perfect for GPU performance

### Type-Safe Generic Code
- Write once, works with any type
- Catch errors at compile time
- No performance penalty

### Real Examples
```cpp
thrust::device_vector<float> d_vec;
thrust::sort(d_vec.begin(), d_vec.end());  // Template-based
```

## Coming Soon

Comprehensive examples including:
- Function and class templates
- Template specialization patterns
- Variadic templates
- Type traits and SFINAE
- GPU-relevant template patterns

## Estimated Time
18-25 hours (challenging but rewarding!)

## Prerequisites
Complete Modules 1-6 first.

**Note**: Templates are complex but essential for modern C++ and GPU programming!