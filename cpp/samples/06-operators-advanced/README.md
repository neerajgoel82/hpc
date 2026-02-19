# Module 6: Operator Overloading and Advanced Features

## Overview
Learn to create intuitive interfaces with operator overloading and explore advanced C++ features. Essential for vector/matrix math in GPU graphics programming.

## Topics Covered

### Operator Overloading
- Arithmetic operators (`+`, `-`, `*`, `/`, etc.)
- Comparison operators (`==`, `!=`, `<`, `>`, etc.)
- Stream operators (`<<`, `>>`)
- Subscript operator (`[]`)
- Function call operator (`()`)
- Assignment operator (`=`)
- Increment/decrement (`++`, `--`)
- Arrow operator (`->`)
- Compound assignment (`+=`, `-=`, etc.)

### Operator Overloading Best Practices
- Member vs non-member operators
- When to return by reference
- Const correctness in operators
- Symmetric operators
- Avoiding surprises (principle of least astonishment)

### Friend Functions and Classes
- Friend functions
- Friend classes
- When to use friends
- Breaking encapsulation carefully

### Conversion Operators
- Implicit vs explicit conversions
- Custom conversion operators
- `explicit` keyword
- Converting constructors

### Functors (Function Objects)
- Overloading `operator()`
- Using functors with STL
- Stateful function objects
- Comparison with lambda functions

### Advanced Features
- Nested classes
- Local classes
- Unions
- Type punning
- Anonymous unions

### Bit Manipulation
- Bitwise operators (`&`, `|`, `^`, `~`, `<<`, `>>`)
- Bit flags and masks
- Bit fields in structs
- Practical bit manipulation

## Why This Matters for GPU

**VERY IMPORTANT FOR GPU:**

### Vector/Matrix Math
```cpp
vec3 a, b, c;
c = a + b;  // Intuitive vector addition
float dot = a * b;  // Dot product
```

### Operator Overloading in GPU Code
- Vector classes (vec2, vec3, vec4)
- Matrix classes (mat3, mat4)
- Quaternions for rotations
- Color operations
- Physics vector operations

### Bit Manipulation
- Packing/unpacking data for GPU
- Optimizing memory usage
- GPU atomic operations often use bitwise ops
- Color format conversions (RGBA packing)

### Examples You'll See
- CUDA libraries use operator overloading heavily
- Graphics math libraries (GLM, Eigen)
- Custom GPU data structures

## Coming Soon

Detailed examples of:
- Complete vector class with operators
- Matrix operations
- Bit manipulation techniques
- Functor examples
- GPU-relevant operator overloading patterns

## Estimated Time
12-15 hours

## Prerequisites
Complete Modules 1-4 first.