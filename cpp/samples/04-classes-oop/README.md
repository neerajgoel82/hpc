# Module 4: Classes and OOP Fundamentals

## Overview
Learn object-oriented programming fundamentals with classes. Understand how to create and manage objects properly.

## Topics Covered

### Classes vs Structs
- When to use class vs struct
- Default access (private vs public)
- GPU programming prefers simple structs

### Class Basics
- Member variables (data members)
- Member functions (methods)
- Access specifiers: public, private, protected
- Encapsulation

### Constructors and Destructors
- Default constructor
- Parameterized constructors
- Copy constructor
- Destructor
- Constructor initialization lists
- Delegating constructors (C++11)

### Special Member Functions
- Copy constructor
- Copy assignment operator
- **Rule of Three**
- When to define them
- `= default` and `= delete` (C++11)

### Advanced Class Features
- `this` pointer
- Const member functions
- Mutable members
- Static members and methods
- Friend functions and classes

### RAII Pattern
- Resource Acquisition Is Initialization
- Automatic resource management
- Exception safety through RAII
- Why RAII matters

## Why This Matters for GPU

- **Structs**: Data structures passed to GPU kernels
- **RAII**: Managing GPU resources (cudaFree in destructors)
- **Object lifetime**: Understanding when memory is allocated/freed
- **Copy semantics**: Understanding data copies before GPU transfer
- **Simple classes**: GPU prefers simple data structures over complex OOP

## Coming Soon

This module is under development with detailed examples of:
- Class design and implementation
- Constructor patterns
- Resource management
- Real-world class examples

## Estimated Time
15-20 hours

## Prerequisites
Complete Modules 1-3 first.