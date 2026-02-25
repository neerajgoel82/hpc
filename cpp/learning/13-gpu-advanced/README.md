# Module 13: Advanced Topics for GPU

## Overview
Learn advanced C++ concepts specifically relevant to GPU programming. These patterns and techniques directly impact GPU performance and code organization.

## Topics Covered

### Data Layout Patterns
- **Array of Structs (AoS)** - Traditional layout
- **Struct of Arrays (SoA)** - GPU-friendly layout
- **AoSoA** - Hybrid approach
- Performance implications
- When to use each
- Conversion between layouts

### Memory Access Patterns
- **Coalesced access** - GPU memory efficiency
- **Strided access** - Performance killer
- Cache line awareness
- Memory alignment
- Padding for alignment
- Bank conflicts in shared memory

### Vectorization
- SIMD concepts
- Compiler auto-vectorization
- Intrinsics (SSE, AVX)
- Vector types
- Alignment requirements
- Vectorization hints

### Alignment and Packing
- Memory alignment rules
- **alignas** specifier (C++11)
- **alignof** operator
- Packed structures
- `#pragma pack`
- Performance impact

### Function Pointers
- Function pointer syntax
- Pointers to member functions
- **std::function** wrapper
- Callbacks and delegates
- Function pointers on GPU (device function pointers)

### Performance-Oriented Design
- Cache-friendly data structures
- Minimizing memory footprint
- False sharing and cache lines
- Branch prediction
- Hot/cold data separation

### Math for GPU
- Vector classes (vec2, vec3, vec4)
- Matrix classes (mat3, mat4)
- Quaternions
- Common math operations
- Optimized math functions

### GPU Memory Patterns
- Pinned (page-locked) memory
- Zero-copy memory
- Unified memory concepts
- Memory pooling
- Custom allocators

### Metaprogramming
- Compile-time computations
- Template metaprogramming basics
- constexpr algorithms
- Type lists
- Generating code at compile time

### GPU-Specific Optimizations
- Occupancy considerations
- Register usage
- Shared memory usage
- Warp divergence
- Branch optimization

## Why This Matters for GPU

### Critical Performance Patterns

#### AoS vs SoA Example
```cpp
// AoS - Bad for GPU (non-coalesced access)
struct Particle {
    float x, y, z;     // position
    float vx, vy, vz;  // velocity
};
Particle particles[N];
// GPU threads access: particles[i].x, particles[i+1].x, etc.
// Non-contiguous in memory!

// SoA - Good for GPU (coalesced access)
struct Particles {
    float* x;   // All x coordinates together
    float* y;   // All y coordinates together
    float* z;   // All z coordinates together
    float* vx;  // All vx together
    float* vy;  // All vy together
    float* vz;  // All vz together
};
// GPU threads access: x[i], x[i+1], x[i+2]
// Contiguous memory! Coalesced access!
```

### Real-World Impact
- AoS vs SoA can mean 10x performance difference
- Proper alignment enables vectorization
- Memory access patterns are #1 GPU performance factor
- These patterns apply to all GPU programming

### Industry Standard
- Game engines use SoA for performance
- Physics simulations require these patterns
- Scientific computing depends on optimal layouts
- All high-performance GPU code uses these techniques

## Coming Soon

Detailed implementations of:
- AoS to SoA conversion utilities
- Aligned memory allocators
- Vector math classes
- Performance comparison examples
- Memory access pattern benchmarks
- Cache-friendly data structure examples
- Real GPU code using these patterns

## Estimated Time
15-20 hours

## Prerequisites
Complete Modules 1-12 first.

**Note**: This module bridges C++ knowledge to GPU performance optimization!