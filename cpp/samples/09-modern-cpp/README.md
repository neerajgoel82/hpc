# Module 9: Modern C++ (C++11/14/17/20)

## Overview
Learn modern C++ features from C++11 through C++20. These features make C++ safer, more expressive, and easier to use. Modern CUDA uses C++14/17 features.

## Topics Covered

### Smart Pointers (C++11)
- **unique_ptr** - Exclusive ownership
- **shared_ptr** - Shared ownership
- **weak_ptr** - Non-owning reference
- **make_unique**, **make_shared**
- Custom deleters
- Converting between smart pointers

### Move Semantics (C++11)
- **Rvalue references** (`&&`)
- **std::move**
- **Move constructors** and **move assignment**
- **Perfect forwarding** with `std::forward`
- **Rule of Five**
- **Rule of Zero**
- When moves happen automatically

### Lambda Functions (C++11)
- Lambda syntax `[](){}`
- Capture clauses `[=]`, `[&]`, `[x, &y]`
- Mutable lambdas
- Generic lambdas (C++14)
- Lambda and GPU kernels
- Init captures (C++14)

### Auto and Type Inference (C++11)
- **auto** keyword
- **decltype** for type deduction
- **auto** with functions
- Advantages and pitfalls of auto
- When to use explicit types

### Range-Based For Loops (C++11)
- Range-for syntax
- Iterating containers
- Reference vs value
- Const iteration

### Uniform Initialization (C++11)
- Brace initialization `{}`
- Preventing narrowing conversions
- List initialization
- Most vexing parse solution

### Nullptr (C++11)
- `nullptr` vs `NULL`
- Type safety
- Overload resolution

### C++17 Features
- **Structured bindings** `auto [x, y] = pair;`
- **if and switch with init** `if (auto x = func(); x > 0)`
- **std::optional** - Maybe value
- **std::variant** - Type-safe union
- **std::any** - Type-safe void*
- **Fold expressions**
- **Inline variables**

### constexpr and consteval
- **constexpr** functions (C++11/14/17 relaxations)
- **constexpr if** (C++17)
- **consteval** immediate functions (C++20)
- **constinit** (C++20)

### Attributes
- `[[nodiscard]]` - Warn if return value ignored
- `[[maybe_unused]]` - Suppress unused warnings
- `[[deprecated]]` - Mark deprecated code
- `[[fallthrough]]` - Intentional fallthrough in switch

## Why This Matters for GPU

### Modern CUDA Uses C++14/17
```cpp
// Smart pointers for automatic cleanup
auto d_ptr = make_unique<float[]>(size);

// Lambdas in thrust
thrust::for_each(data.begin(), data.end(),
    [](float& x) { x = x * 2.0f; });

// Auto for complex types
auto result = thrust::reduce(data.begin(), data.end());
```

### Benefits for GPU Code
- **Smart pointers**: RAII for CUDA memory
- **Move semantics**: Efficient data transfers
- **Lambdas**: Inline GPU operations (thrust)
- **Auto**: Simplify complex GPU template types
- **constexpr**: Compile-time GPU configuration

### Real CUDA Examples
- Thrust library uses lambdas extensively
- Modern CUDA APIs use smart pointers
- Move semantics reduce unnecessary copies
- constexpr for grid/block dimensions

## Coming Soon

Comprehensive coverage of:
- Smart pointer patterns
- Move semantics in depth
- Lambda expressions and captures
- All C++17/20 features with examples
- GPU-specific modern C++ usage

## Estimated Time
20-30 hours

## Prerequisites
Complete Modules 1-8 first.

**Note**: These features are essential for modern C++ and GPU programming!