# C++ Learning Path for GPU Programming

A comprehensive, hands-on curriculum covering C++ fundamentals through GPU programming with CUDA. This repository provides 14 structured modules with practical examples designed to take you from C++ basics to writing efficient GPU code.

## 🎯 What You'll Learn

- **Complete C++ fundamentals** - Syntax, types, control flow, functions
- **Object-Oriented Programming** - Classes, inheritance, polymorphism
- **Modern C++** - C++11/14/17/20 features, smart pointers, lambdas
- **Memory management** - Pointers, references, dynamic allocation
- **Templates and STL** - Generic programming, containers, algorithms
- **Concurrency** - Multithreading, atomics, parallel patterns
- **Build systems** - Makefiles, CMake, debugging, profiling
- **GPU-specific concepts** - Memory layouts, optimization patterns
- **CUDA programming** - Your first GPU programs

## 📚 Curriculum Overview

### **Fundamentals (Modules 1-3)**
- Module 1: C++ Fundamentals
- Module 2: Functions and Program Structure
- Module 3: Pointers and Memory

### **Object-Oriented Programming (Modules 4-6)**
- Module 4: Classes and OOP Fundamentals
- Module 5: Inheritance and Polymorphism
- Module 6: Operator Overloading

### **Advanced C++ (Modules 7-10)**
- Module 7: Templates and Generic Programming
- Module 8: STL and Standard Library
- Module 9: Modern C++ (C++11/14/17/20)
- Module 10: Exception Handling

### **Parallel Programming (Modules 11-14)**
- Module 11: Multithreading and Concurrency
- Module 12: Build Systems, Debugging & Tools
- Module 13: Advanced Topics for GPU
- Module 14: GPU Programming with CUDA

## 🚀 Quick Start

### Prerequisites
- macOS, Linux, or Windows with WSL
- C++ compiler (g++ or clang)
- VS Code or Cursor IDE (optional but recommended)

### Getting Started
```bash
# Clone this repository
cd cpp-samples

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

## 📖 How to Use This Repository

1. **Read LEARNING_PATH.md** - Detailed curriculum overview
2. **Start with Module 1** - Complete modules sequentially
3. **Read the code** - Every example is heavily commented
4. **Experiment** - Try the "TRY THIS" suggestions in each file
5. **Practice** - Complete exercises at the end of each module

## 🎓 Learning Approach

For each module:
1. **Read** example code and comments
2. **Predict** what the code will do
3. **Compile and run** to verify
4. **Modify** the code to experiment
5. **Complete** practice exercises
6. **Move forward** when comfortable

## ⏱️ Estimated Timeline

- **Modules 1-3**: 4-6 weeks (fundamentals and memory)
- **Modules 4-6**: 4-5 weeks (OOP)
- **Modules 7-10**: 5-7 weeks (advanced C++)
- **Modules 11-14**: 5-6 weeks (parallel and GPU)

**Total**: 18-24 weeks for comprehensive mastery

*Adjust based on your pace and prior experience*

## 🎯 Current Status

- ✅ **Module 1**: Complete (9 examples)
- 🚧 **Modules 2-14**: READMEs created, examples coming soon

## 📋 Module Completion Checklist

Track your progress:
- [x] Module 1: C++ Fundamentals ← **START HERE**
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

## 💡 Why This Curriculum?

### GPU-Focused Approach
Every concept is taught with GPU programming in mind:
- Data types matter for GPU performance
- Memory management is critical for CPU↔GPU transfers
- Understanding parallelism prepares you for GPU threads
- Build systems are essential for CUDA projects

### Comprehensive Coverage
- All core C++ concepts
- Complete OOP knowledge
- Modern C++ features
- Real-world tools and practices
- Direct path to GPU programming

### Hands-On Learning
- 100+ code examples
- Practical exercises
- Real-world patterns
- Performance-oriented

## 🛠️ Tools You'll Master

- **Compilers**: g++, clang, nvcc (CUDA)
- **Build Systems**: Make, CMake
- **Debuggers**: lldb, gdb, cuda-gdb
- **Profilers**: gprof, Nsight, Instruments
- **Memory Tools**: Valgrind, AddressSanitizer
- **Testing**: Google Test

## 📚 Resources

- [C++ Reference](https://cppreference.com) - Comprehensive C++ documentation
- [CUDA Documentation](https://docs.nvidia.com/cuda/) - Official CUDA guides
- [Modern C++ Features](https://github.com/AnthonyCalandra/modern-cpp-features) - C++11/14/17/20 reference

## 🤝 Contributing

Found an issue or have a suggestion? Feel free to open an issue or submit a pull request.

## 📄 License

This educational repository is provided as-is for learning purposes.

---

**Ready to begin?** Start with [Module 1: C++ Fundamentals](01-basics/) or read the complete [LEARNING_PATH.md](LEARNING_PATH.md) for detailed information.

Good luck on your journey from C++ to GPU programming! 🚀
