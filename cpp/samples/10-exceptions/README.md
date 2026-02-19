# Module 10: Exception Handling and Error Management

## Overview
Learn C++ exception handling and error management strategies. Understand when to use exceptions vs error codes, especially important for GPU programming.

## Topics Covered

### Exception Basics
- **try/catch/throw** syntax
- Exception propagation
- Stack unwinding
- Catching by value, reference, or pointer
- Catch-all handler `catch(...)`

### Exception Classes
- **std::exception** hierarchy
- **std::runtime_error**
- **std::logic_error**
- **std::out_of_range**
- **std::bad_alloc**
- Other standard exceptions

### Custom Exceptions
- Creating exception classes
- Inheriting from std::exception
- `what()` method
- Best practices for custom exceptions

### Exception Safety
- **No-throw guarantee** - Never throws
- **Strong guarantee** - Atomic (commit or rollback)
- **Basic guarantee** - Valid state but not original
- **No guarantee** - May leave invalid state

### noexcept Specifier (C++11)
- **noexcept** keyword
- `noexcept(expression)`
- Benefits of noexcept
- Move constructors and noexcept
- Performance implications

### RAII and Exception Safety
- Resource cleanup with RAII
- Destructors and exceptions
- Smart pointers for exception safety
- Lock guards and exception safety

### Advanced Exception Handling
- Rethrowing exceptions
- **std::exception_ptr** (C++11)
- **std::current_exception**
- **std::rethrow_exception**
- Nested exceptions (C++11)

### Error Handling Strategies
- Exceptions vs error codes
- When to use each
- Performance considerations
- Error handling in libraries
- Error handling best practices

### assert and static_assert
- **assert** for runtime checks
- **static_assert** for compile-time checks
- When to use assertions
- Release vs debug builds

## Why This Matters for GPU

### GPU Does NOT Use Exceptions
**Important**: CUDA device code (GPU kernels) **cannot use exceptions**
- No try/catch in GPU kernels
- No exception propagation on device
- GPU uses error codes instead

### CPU-Side Exception Handling
```cpp
try {
    // Allocate GPU memory
    cudaMalloc(&d_ptr, size);
    if (cudaGetLastError() != cudaSuccess) {
        throw std::runtime_error("GPU allocation failed");
    }

    // Launch kernel
    kernel<<<grid, block>>>(d_ptr);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}
catch (const std::exception& e) {
    // Clean up
    std::cerr << "GPU error: " << e.what() << std::endl;
    cudaFree(d_ptr);
    throw;  // Rethrow
}
```

### CUDA Error Handling Pattern
- Check CUDA API return codes
- Wrap CUDA calls in error-checking functions
- Use RAII for GPU memory management
- Convert CUDA errors to exceptions on host

### Best Practices for GPU Code
- Use exceptions in CPU/host code
- Check CUDA error codes explicitly
- RAII for GPU resource management
- Never assume GPU operations succeed

## Coming Soon

Detailed examples including:
- Exception handling patterns
- Custom exception classes
- RAII and exception safety
- CUDA error handling wrappers
- When to use exceptions vs error codes

## Estimated Time
10-15 hours

## Prerequisites
Complete Modules 1-9 first.

**Key Takeaway**: Understand exceptions thoroughly, but remember GPU kernels can't use them!