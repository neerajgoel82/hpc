# Module 5: Writing Your First Kernels

**Part of**: FreeCodeCamp CUDA Programming Course
**Duration**: 8-12 hours
**Difficulty**: Beginner to Intermediate

---

## Overview

Module 5 introduces CUDA kernel programming from the ground up. You'll learn thread indexing, write vector and matrix operations, understand profiling basics, use atomic operations, and master CUDA streams for asynchronous execution.

---

## Notebooks in This Module

### 01. CUDA Basics: Thread Indexing (`01_CUDA_Basics_01_idxing.ipynb`)
**Duration**: 1-1.5 hours

**Learning Objectives**:
- Understand CUDA thread hierarchy (grids, blocks, threads)
- Calculate global thread IDs in 1D, 2D, 3D
- Master `blockIdx`, `threadIdx`, `blockDim`, `gridDim`
- Handle boundary conditions

**Key Concepts**:
- Thread indexing formula: `idx = blockIdx.x * blockDim.x + threadIdx.x`
- 2D indexing for matrices
- Grid stride loops
- Bounds checking

**What You'll Build**: Index calculation kernels with visualization

---

### 02. Kernels: Vector Addition v1 (`02_Kernels_00_vector_add_v1.ipynb`)
**Duration**: 1 hour

**Learning Objectives**:
- Write your first CUDA kernel
- Allocate GPU memory with `cudaMalloc`
- Transfer data with `cudaMemcpy`
- Launch kernels with `<<<>>>` syntax

**Key Concepts**:
- Kernel definition with `__global__`
- Memory management lifecycle
- Host-to-device and device-to-host transfers
- Error checking basics

**What You'll Build**: Simple vector addition kernel

---

### 03. Kernels: Vector Addition v2 (`02_Kernels_01_vector_add_v2.ipynb`)
**Duration**: 1 hour

**Learning Objectives**:
- Optimize vector addition
- Use grid stride loops for large arrays
- Handle arrays larger than thread count
- Implement proper error checking

**Key Concepts**:
- Grid stride pattern: `for (int i = idx; i < n; i += stride)`
- Flexible kernel sizing
- Comprehensive error handling
- Performance measurement

**What You'll Build**: Optimized vector addition with grid stride

---

### 04. Kernels: Matrix Multiplication (`02_Kernels_02_matmul.ipynb`)
**Duration**: 2 hours

**Learning Objectives**:
- Implement naive matrix multiplication
- Work with 2D thread blocks
- Understand row-major memory layout
- Calculate 2D indices correctly

**Key Concepts**:
- 2D thread organization
- Matrix indexing: `C[row][col] = row * width + col`
- O(n³) algorithm implementation
- Memory access patterns

**What You'll Build**: Basic matrix multiplication kernel

**Performance**: ~0.1-0.5 TFLOPS (naive implementation)

---

### 05. Profiling: NVTX Markers (`03_Profiling_00_nvtx_matmul.ipynb`)
**Duration**: 1 hour

**Learning Objectives**:
- Use NVTX for code annotation
- Mark regions in profiling timeline
- Understand profiling workflow
- Integrate with Nsight Systems

**Key Concepts**:
- NVTX ranges and markers
- Color-coded timeline regions
- Performance hotspot identification
- Profiling best practices

**Tools**: NVTX API, Nsight Systems

---

### 06. Profiling: Naive Matrix Multiply (`03_Profiling_01_naive_matmul.ipynb`)
**Duration**: 1.5 hours

**Learning Objectives**:
- Profile matrix multiplication performance
- Identify memory bandwidth bottlenecks
- Measure kernel execution time
- Understand performance metrics

**Key Concepts**:
- Effective bandwidth calculation
- FLOPS (floating-point operations per second)
- Memory throughput analysis
- Profiling with CUDA events

**What You'll Analyze**: Why naive matmul is slow

---

### 07. Profiling: Tiled Matrix Multiply (`03_Profiling_02_tiled_matmul.ipynb`)
**Duration**: 2-3 hours

**Learning Objectives**:
- Implement tiled matrix multiplication
- Use shared memory for optimization
- Achieve 10-20x speedup
- Understand memory hierarchy benefits

**Key Concepts**:
- Shared memory tiling strategy
- Memory reuse patterns
- `__syncthreads()` for synchronization
- Tile size optimization

**What You'll Build**: Optimized tiled matmul with shared memory

**Performance**: 2-5 TFLOPS (10-20x faster than naive)

---

### 08. Atomics: Atomic Addition (`04_Atomics_00_atomicAdd.ipynb`)
**Duration**: 1.5 hours

**Learning Objectives**:
- Use `atomicAdd` for thread-safe updates
- Understand atomic operation overhead
- Implement parallel reduction with atomics
- Learn optimization strategies

**Key Concepts**:
- Atomic operations basics
- Race conditions and how atomics prevent them
- Atomic performance costs
- When to use atomics vs other approaches

**What You'll Build**: Histogram, sum reduction with atomics

---

### 09. Streams: Stream Basics (`05_Streams_01_stream_basics.ipynb`)
**Duration**: 1.5 hours

**Learning Objectives**:
- Create and use CUDA streams
- Overlap computation and memory transfers
- Understand asynchronous execution
- Manage stream synchronization

**Key Concepts**:
- Stream creation and destruction
- Asynchronous kernel launches
- Asynchronous memory copies (requires pinned memory)
- Stream synchronization

**What You'll Build**: Async data pipeline with streams

**Performance**: 2-3x speedup with overlap

---

### 10. Streams: Advanced Patterns (`05_Streams_02_stream_advanced.ipynb`)
**Duration**: 2 hours

**Learning Objectives**:
- Implement multi-stream pipelines
- Use CUDA events for dependencies
- Build producer-consumer patterns
- Maximize GPU utilization

**Key Concepts**:
- Multiple concurrent streams
- Event-based synchronization
- Pipeline patterns
- Deep pipelining (3+ stages)

**What You'll Build**: Multi-stage async processing pipeline

---

## Learning Path

```
01_idxing → 02_vector_add_v1 → 02_vector_add_v2
                    ↓
            02_matmul (2D indexing)
                    ↓
    03_profiling_nvtx → 03_naive_matmul → 03_tiled_matmul
                    ↓
            04_atomics
                    ↓
        05_streams_basic → 05_streams_advanced
```

## Prerequisites

### Knowledge
- Python programming
- Basic C/C++ syntax
- Understanding of arrays and matrices
- Basic parallel computing concepts

### Setup
- Google Colab account
- Web browser
- Internet connection
- **No local GPU required!**

## Success Criteria

By completing Module 5, you should be able to:
- [ ] Write CUDA kernels from scratch
- [ ] Calculate thread indices in 1D, 2D, 3D
- [ ] Manage GPU memory allocation and transfers
- [ ] Implement vector and matrix operations
- [ ] Use shared memory for optimization
- [ ] Profile CUDA applications
- [ ] Use atomic operations correctly
- [ ] Create asynchronous pipelines with streams
- [ ] Achieve 10-20x speedup with optimization

## Key Concepts Summary

### Thread Indexing
```cuda
// 1D
int idx = blockIdx.x * blockDim.x + threadIdx.x;

// 2D
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

### Memory Management
```cuda
// Allocate
cudaMalloc(&d_ptr, size);

// Copy to GPU
cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice);

// Copy from GPU
cudaMemcpy(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost);

// Free
cudaFree(d_ptr);
```

### Shared Memory Tiling
```cuda
__shared__ float tile[TILE_SIZE][TILE_SIZE];

// Load tile
tile[ty][tx] = data[row][col];
__syncthreads();

// Compute using tile
result += tile[ty][k] * ...;
__syncthreads();
```

### Streams
```cuda
cudaStream_t stream;
cudaStreamCreate(&stream);

// Async operations
cudaMemcpyAsync(d_data, h_data, size, H2D, stream);
kernel<<<grid, block, 0, stream>>>();
cudaStreamSynchronize(stream);
```

## Performance Expectations

| Operation | Naive | Optimized | Speedup |
|-----------|-------|-----------|---------|
| **Vector Add** | Baseline | Grid stride | 1-2x |
| **Matrix Multiply** | 0.1-0.5 TFLOPS | 2-5 TFLOPS | 10-20x |
| **With Streams** | Baseline | Async pipeline | 2-3x |

## Common Pitfalls

1. **Forgetting Bounds Checking**
   ```cuda
   // Always check!
   if (idx < n) {
       data[idx] = value;
   }
   ```

2. **Missing __syncthreads()**
   ```cuda
   // Load to shared memory
   tile[tid] = data[idx];
   __syncthreads();  // MUST sync before reading!
   // Now safe to read tile[]
   ```

3. **Wrong Memory Copy Direction**
   ```cuda
   cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice);  // H→D
   cudaMemcpy(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost);  // D→H
   ```

4. **Atomic Overuse**
   ```cuda
   // Atomics are slow - use sparingly
   // Consider reduction or shared memory first
   ```

## Time Estimate
- **Fast pace**: 1 week (8-10 hours)
- **Moderate pace**: 1.5 weeks (6-8 hours/week)
- **Relaxed pace**: 2 weeks (4-6 hours/week)

## Additional Resources

### Official Documentation
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Nsight Systems Guide](https://docs.nvidia.com/nsight-systems/)

### Course Resources
- [FreeCodeCamp CUDA Course (YouTube)](https://www.youtube.com/watch?v=86FAWCzIe_4)
- [Original Course Repository](https://github.com/Infatoshi/cuda-course)

### Recommended Reading
- "CUDA by Example" - Sanders & Kandrot
- "Programming Massively Parallel Processors" - Hwu et al.

## Practice Exercises

1. **Vector operations**: Implement dot product, scalar multiply
2. **Matrix operations**: Transpose, element-wise multiply
3. **Optimization challenge**: Improve tiled matmul further
4. **Atomic reductions**: Sum, max, min with atomics
5. **Stream pipeline**: 4-stage processing pipeline

## Troubleshooting

### GPU Not Available
**Solution**: Runtime → Change runtime type → T4 GPU → Save

### Kernel Doesn't Run
**Solution**: Check error with `cudaGetLastError()`

### Slow Performance
**Solution**: Profile with CUDA events, check memory bandwidth

### Wrong Results
**Solution**: Verify thread indexing, check bounds, validate on small inputs

## Next Module

After completing Module 5, proceed to:
**Module 6: CUDA APIs** - Learn to use cuBLAS, cuDNN, and other optimized libraries for production performance.

**Path**: `../module6/README.md`

---

**Pro Tip**: Module 5 is foundational. Master thread indexing and shared memory - you'll use these everywhere in CUDA programming!

## Questions to Test Understanding

1. How do you calculate a global thread ID in 2D?
2. Why is tiled matrix multiplication faster than naive?
3. When must you call `__syncthreads()`?
4. What's the difference between `cudaMemcpy` and `cudaMemcpyAsync`?
5. Why are atomic operations slower than regular operations?
6. How do streams enable overlapping computation and transfers?
7. What is grid stride and when should you use it?
8. How do you measure kernel execution time accurately?

If you can answer these and implement tiled matmul from scratch, you're ready for Module 6! 🚀
