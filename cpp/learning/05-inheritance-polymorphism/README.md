# Module 5: Inheritance and Polymorphism

## Overview
Complete your OOP education with inheritance, polymorphism, and advanced type casting. Learn why GPU programming often avoids these features.

## Topics Covered

### Inheritance
- Base and derived classes
- Protected access specifier
- Inheritance syntax
- Constructor/destructor call order
- **Access control**: public, protected, private inheritance
- Single vs multiple inheritance
- **Diamond problem** and virtual inheritance

### Polymorphism
- Function overriding
- **Virtual functions** and vtables
- Runtime polymorphism
- Compile-time vs runtime binding

### Abstract Classes
- **Pure virtual functions** (`= 0`)
- Abstract base classes
- Interfaces in C++
- When to use abstractions

### Virtual Destructors
- Why virtual destructors are critical
- Memory leaks without virtual destructors
- Rule: Base class with virtual functions needs virtual destructor

### Object Slicing
- What is object slicing?
- How to avoid it
- Storing derived objects in base pointers

### Type Casting
- **static_cast** - Compile-time type conversion
- **dynamic_cast** - Runtime type checking (requires RTTI)
- **const_cast** - Cast away const
- **reinterpret_cast** - Low-level bit pattern reinterpretation
- C-style casts (avoid in modern C++)

### RTTI (Run-Time Type Information)
- `typeid` operator
- `type_info` class
- When RTTI is useful
- Performance cost of RTTI

## Why This Matters for GPU

**Important to understand, but...**

GPU programming typically **avoids** inheritance and virtual functions because:
- Virtual function calls require vtable lookups (slow on GPU)
- Dynamic dispatch doesn't parallelize well
- GPUs prefer simple data structures
- Templates (Module 7) are preferred over runtime polymorphism

**However, you'll still encounter:**
- Inheritance in CPU-side code (host code)
- Template-based compile-time polymorphism
- Type casting when working with GPU pointers
- Understanding OOP helps design better host-side APIs

## Coming Soon

Comprehensive examples including:
- Inheritance hierarchies
- Virtual function mechanics
- Abstract classes and interfaces
- Type casting scenarios
- Why GPU avoids these patterns

## Estimated Time
15-20 hours

## Prerequisites
Complete Modules 1-4 first.