# Phase 1: Foundations (Weeks 1-2)

## Overview
Phase 1 introduces CUDA programming fundamentals, covering GPU architecture basics and the thread hierarchy model.

## Notebooks in This Phase

### 00-setup-verification.ipynb ⭐ START HERE
**Duration**: 15-20 minutes
**Learning Objectives**:
- Verify CUDA installation in Colab
- Query GPU properties
- Run your first CUDA kernel
- Understand error checking

**Key Concepts**:
- GPU vs CPU architecture
- CUDA runtime API
- Kernel launch syntax `<<<grid, block>>>`
- Thread identification

---

### 01-hello-world.ipynb
**Duration**: 30 minutes
**Learning Objectives**:
- Write basic CUDA kernels
- Experiment with different launch configurations
- Understand `__global__` qualifier
- Practice with built-in variables

**Key Concepts**:
- Kernel functions
- Launch configuration
- `threadIdx`, `blockIdx`, `blockDim`, `gridDim`

---

### 02-device-query.ipynb
**Duration**: 45 minutes
**Learning Objectives**:
- Deep dive into GPU architecture
- Understand compute capability
- Learn about memory hierarchy
- Understand multiprocessors (SMs)

**Key Concepts**:
- Streaming Multiprocessors (SMs)
- Warps (32 threads)
- Memory types (global, shared, registers)
- Compute capability

---

### 03-thread-indexing.ipynb
**Duration**: 1 hour
**Learning Objectives**:
- Master 1D, 2D, and 3D thread indexing
- Calculate global thread IDs
- Handle boundary conditions
- Optimize thread configurations

**Key Concepts**:
- 1D indexing: `blockIdx.x * blockDim.x + threadIdx.x`
- 2D indexing for matrices
- 3D indexing for volumes
- Grid stride loops

**Practice Problems**:
- Calculate global ID for various configurations
- Convert 2D coordinates to 1D index
- Handle non-power-of-2 array sizes

---

### 04-vector-operations.ipynb
**Duration**: 1.5 hours
**Learning Objectives**:
- Implement vector addition on GPU
- Compare CPU vs GPU performance
- Understand memory transfer overhead
- Practice proper error checking

**Key Concepts**:
- `cudaMalloc` / `cudaFree`
- `cudaMemcpy`
- Host-device synchronization
- Performance measurement with CUDA events

**Implementations**:
- Vector addition
- Vector scalar multiplication
- Dot product (reduce result)

---

### 05-matrix-operations.ipynb
**Duration**: 1.5 hours
**Learning Objectives**:
- Work with 2D data structures
- Use 2D thread blocks and grids
- Implement matrix addition
- Understand memory layout (row-major)

**Key Concepts**:
- 2D thread blocks
- Matrix indexing in CUDA
- Row-major vs column-major layout
- Coalesced memory access introduction

**Implementations**:
- Matrix addition
- Matrix transpose (naive version)
- Matrix-vector multiplication

---

## Learning Path

```
00-setup-verification
        ↓
01-hello-world
        ↓
02-device-query
        ↓
03-thread-indexing
        ↓
04-vector-operations
        ↓
05-matrix-operations
```

## Prerequisites
- Strong C/C++ knowledge
- Understanding of pointers and memory management
- Basic knowledge of parallel computing concepts (helpful)

## Success Criteria

By the end of Phase 1, you should be able to:
- [ ] Write and launch CUDA kernels
- [ ] Calculate global thread IDs for 1D, 2D, and 3D grids
- [ ] Allocate and transfer memory between host and device
- [ ] Implement simple parallel algorithms (vector/matrix operations)
- [ ] Check for CUDA errors properly
- [ ] Measure kernel execution time
- [ ] Explain GPU architecture basics (SMs, warps, threads)

## Common Pitfalls

1. **Forgetting to sync after kernel launch**
   ```cuda
   kernel<<<grid, block>>>();
   cudaDeviceSynchronize();  // Don't forget!
   ```

2. **Not checking for errors**
   ```cuda
   cudaError_t err = cudaMalloc(&d_ptr, size);
   if (err != cudaSuccess) {
       printf("Error: %s\n", cudaGetErrorString(err));
   }
   ```

3. **Memory leaks**
   ```cuda
   cudaMalloc(&d_ptr, size);
   // ... use d_ptr ...
   cudaFree(d_ptr);  // Always free!
   ```

4. **Incorrect thread indexing**
   ```cuda
   // Correct:
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx < n) {  // Boundary check!
       array[idx] = ...;
   }
   ```

5. **Wrong memory copy direction**
   ```cuda
   cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);  // CPU → GPU
   cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);  // GPU → CPU
   ```

## Time Estimate
- **Fast pace**: 1 week (2-3 hours/day)
- **Moderate pace**: 2 weeks (1 hour/day)
- **Relaxed pace**: 3 weeks (30 min/day)

## Additional Resources

### Videos
- NVIDIA CUDA Training Series (YouTube)
- "Intro to Parallel Programming" (Udacity)

### Reading
- CUDA C Programming Guide - Chapters 1-3
- "CUDA by Example" - Chapters 1-4

### Practice
- Modify the notebooks with different launch configurations
- Try different array sizes
- Benchmark CPU vs GPU performance
- Experiment with error conditions

## Next Phase

Once comfortable with Phase 1, move to:
**Phase 2: Memory Management** - Learn about different memory types, optimization techniques, and shared memory.

---

**Pro Tip**: Don't rush through Phase 1. A solid understanding of the fundamentals will make advanced topics much easier!

## Questions to Test Your Understanding

1. What is the difference between `__global__`, `__device__`, and `__host__`?
2. How many threads are in a warp?
3. What is the maximum number of threads per block?
4. Why do we need `cudaDeviceSynchronize()`?
5. How do you calculate a global thread ID in a 2D grid?
6. What happens if you launch more threads than array elements?
7. When should you check for CUDA errors?

If you can answer all these confidently, you're ready for Phase 2!
