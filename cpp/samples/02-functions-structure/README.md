# Module 2: Functions and Program Structure

## Overview
Learn how to organize C++ code properly with functions, headers, and the preprocessor. This module teaches you how real C++ projects are structured.

## Topics Covered

### Function Parameters
- Pass by value
- Pass by reference
- Pass by pointer
- When to use each
- Const parameters

### Function Features
- Function overloading
- Default arguments
- Inline functions
- Return value optimization

### Code Organization
- **Header files (.h)** - Declarations
- **Implementation files (.cpp)** - Definitions
- Header guards
- `#pragma once`
- Include paths

### Preprocessor
- `#include` directive
- `#define` macros
- `#ifdef`, `#ifndef`, `#endif`
- Conditional compilation
- `__FILE__`, `__LINE__` macros
- When to use (and not use) macros

### Initialization
- Direct initialization
- Copy initialization
- List initialization `{}`
- Aggregate initialization
- Uniform initialization (C++11)

### File I/O
- Reading text files
- Writing text files
- `ifstream`, `ofstream`, `fstream`
- Error handling with files
- File positioning

## Why This Matters for GPU

- **Header files**: CUDA code uses .cu files with .h/.cuh headers
- **Preprocessor**: Conditional compilation for device/host code
- **Function organization**: Separating kernel declarations from implementations
- **File I/O**: Loading data for GPU processing

## Coming Soon

This module is under development. It will include:
- Example programs with proper header/implementation separation
- Preprocessor macro examples
- File I/O with real data files
- Build examples showing compilation of multiple files

## Estimated Time
10-15 hours

## Prerequisites
Complete Module 1 first.