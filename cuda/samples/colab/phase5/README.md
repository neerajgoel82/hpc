# Phase 5: Advanced Algorithms (Weeks 9-10)

## Overview
Phase 5 focuses on production-quality algorithm implementations. You'll optimize matrix multiplication to near-peak performance, integrate CUDA libraries, and master fundamental algorithms like sorting.

## Notebooks in This Phase

### 24_optimized_gemm.ipynb ⭐ KEY NOTEBOOK
**Duration**: 3-4 hours
**Learning Objectives**:
- Implement highly optimized matrix multiply (GEMM)
- Apply all optimization techniques learned
- Achieve 60-80% of theoretical peak performance
- Understand cuBLAS implementation strategies

**Key Concepts**:
- Multiple levels of tiling
- Register blocking
- Thread coarsening
- Memory access optimization
- Instruction-level parallelism

**Implementations**:
- Naive GEMM
- Tiled GEMM (Phase 2 review)
- Register-blocked GEMM
- Fully optimized GEMM with all techniques

**Performance**: 2-5 TFLOPS on modern GPUs

---

### 25_cublas_integration.ipynb
**Duration**: 2 hours
**Learning Objectives**:
- Use cuBLAS library for matrix operations
- Compare custom kernels vs cuBLAS
- Understand BLAS API conventions
- Integrate cuBLAS into applications

**Key Concepts**:
- cuBLAS handle management
- Column-major vs row-major layout
- GEMM, GEMV operations
- Batched operations
- cuBLAS performance tuning

**API Coverage**:
- `cublasSgemm` - Single precision GEMM
- `cublasDgemm` - Double precision GEMM
- `cublasSgemv` - Matrix-vector multiply
- `cublasSnrm2` - Vector norm

---

### 26_matrix_transpose.ipynb
**Duration**: 2 hours
**Learning Objectives**:
- Implement efficient matrix transpose
- Eliminate shared memory bank conflicts
- Optimize for memory bandwidth
- Achieve peak transpose performance

**Key Concepts**:
- Naive transpose (uncoalesced)
- Shared memory transpose
- Bank conflict elimination with padding
- Rectangular vs square matrices

**Implementations**:
- Naive transpose (slow)
- Shared memory transpose (fast)
- Conflict-free transpose with padding (fastest)

**Performance**: 200-300 GB/s bandwidth

---

### 27_bitonic_sort.ipynb
**Duration**: 2 hours
**Learning Objectives**:
- Implement bitonic sorting network
- Understand comparison-based parallel sorting
- Use shared memory for fast local sorting
- Learn when to use bitonic vs other algorithms

**Key Concepts**:
- Sorting networks
- Bitonic sequence properties
- Parallel comparison-exchange
- O(n log² n) complexity
- Best for small-medium arrays

**Implementations**:
- Bitonic merge
- Bitonic sort kernel
- Multi-block bitonic sort

---

### 28_radix_sort.ipynb
**Duration**: 2.5 hours
**Learning Objectives**:
- Implement radix sort on GPU
- Use prefix sum (scan) for sorting
- Achieve O(n) integer sorting
- Compare with CPU quicksort

**Key Concepts**:
- Radix sort algorithm
- Digit-wise sorting (4-8 bits per pass)
- Scan for computing positions
- Multi-pass strategy
- O(k·n) where k = #passes

**Implementations**:
- 1-bit radix sort
- 4-bit radix sort (optimized)
- Comparison with thrust::sort

**Performance**: 1-2 billion keys/second

---

### 29_thrust_examples.ipynb
**Duration**: 2 hours
**Learning Objectives**:
- Use Thrust library (STL for CUDA)
- Write high-level GPU code quickly
- Understand Thrust algorithms
- When to use Thrust vs custom kernels

**Key Concepts**:
- Thrust vectors (device/host)
- Thrust algorithms: transform, reduce, scan, sort
- Custom functors
- Iterators and ranges
- Performance vs ease-of-use trade-offs

**Examples**:
- Vector operations with thrust::transform
- Reduction with thrust::reduce
- Sorting with thrust::sort
- Stream compaction with thrust::copy_if

---

## Learning Path

```
24-optimized-gemm ⭐
        ↓
25-cublas-integration
        ↓
26-matrix-transpose
        ↓
27-bitonic-sort
        ↓
28-radix-sort
        ↓
29-thrust-examples
```

## Prerequisites
- Completed Phases 1-4
- Strong algorithm design skills
- Understanding of all optimization techniques
- Comfort with complex implementations

## Success Criteria

By the end of Phase 5, you should be able to:
- [ ] Implement highly optimized GEMM from scratch
- [ ] Use cuBLAS effectively in applications
- [ ] Write bank-conflict-free matrix transpose
- [ ] Implement bitonic and radix sort
- [ ] Use Thrust library for rapid development
- [ ] Achieve production-level performance
- [ ] Benchmark and compare implementations
- [ ] Choose the right algorithm for the problem

## Performance Targets

Target performance on modern GPUs (RTX 3090, A100):

| Algorithm | Performance | Notes |
|-----------|-------------|-------|
| **Optimized GEMM** | 15-25 TFLOPS (FP32) | 60-80% of cuBLAS |
| **cuBLAS GEMM** | 20-30 TFLOPS (FP32) | Near-peak performance |
| **Matrix Transpose** | 200-300 GB/s | Bandwidth-bound |
| **Radix Sort** | 1-2 B keys/sec | Integer sorting |
| **Bitonic Sort** | 100-500 M keys/sec | Small arrays only |
| **Thrust Sort** | 1-2 B keys/sec | Comparable to custom |

## Key Algorithms

### 1. GEMM Optimization Stages
```
Naive GEMM (0.1 TFLOPS)
    ↓ Add tiling
Tiled GEMM (2-3 TFLOPS)
    ↓ Add register blocking
Register-blocked GEMM (5-10 TFLOPS)
    ↓ Add vectorization + ILP
Fully Optimized GEMM (15-25 TFLOPS)
```

### 2. Transpose Optimization
```cuda
// Conflict-free with padding
#define TILE_DIM 32
#define BLOCK_ROWS 8
__shared__ float tile[TILE_DIM][TILE_DIM+1]; // +1 to avoid conflicts

// Load with coalescing
tile[threadIdx.y][threadIdx.x] = in[index_in];
__syncthreads();

// Store with coalescing (transposed)
out[index_out] = tile[threadIdx.x][threadIdx.y];
```

### 3. Radix Sort Pattern
```
For each digit (4 bits):
  1. Extract digit
  2. Compute histogram (count per bin)
  3. Exclusive scan (prefix sum) → positions
  4. Scatter to sorted positions
  5. Repeat for next digit
```

## Common Pitfalls

1. **GEMM Inefficiency**
   ```cuda
   // Must use:
   // - Multiple levels of tiling
   // - Register blocking (compute 4x4 or 8x8 tiles per thread)
   // - Vectorized loads (float4)
   // - Maximize instruction-level parallelism
   ```

2. **cuBLAS Layout Confusion**
   ```cpp
   // cuBLAS uses column-major (Fortran) layout!
   // For row-major C, use: C = B^T * A^T instead of C = A * B
   cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
               m, n, k, &alpha, B, ldb, A, lda, &beta, C, ldc);
   ```

3. **Bank Conflicts in Transpose**
   ```cuda
   // BAD - bank conflicts
   __shared__ float tile[32][32];
   
   // GOOD - add padding
   __shared__ float tile[32][33];  // Extra column eliminates conflicts
   ```

4. **Radix Sort Overflow**
   ```cuda
   // Must handle partial blocks correctly
   // Use proper scan implementation
   // Handle last block with fewer elements
   ```

## Optimization Techniques Summary

### Learned in Phase 5:
- **Register blocking** - Each thread computes multiple outputs
- **Vectorized loads** - Use float4 for 4x coalescing
- **Loop unrolling** - Reduce loop overhead
- **Instruction-level parallelism** - Keep pipeline full
- **Bank conflict elimination** - Padding shared memory

### All Techniques (Phases 1-5):
1. Memory coalescing (Phase 2)
2. Shared memory tiling (Phase 2)
3. Warp-level operations (Phase 3)
4. Occupancy tuning (Phase 3)
5. Texture/constant memory (Phase 4)
6. Register blocking (Phase 5)
7. Vectorization (Phase 5)

## Time Estimate
- **Fast pace**: 2 weeks (3-4 hours/day)
- **Moderate pace**: 2.5 weeks (2-3 hours/day)
- **Relaxed pace**: 3-4 weeks (1-2 hours/day)

## Additional Resources

### NVIDIA Documentation
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [Thrust Quick Start Guide](https://docs.nvidia.com/cuda/thrust/)
- [CUTLASS (CUDA Templates for Linear Algebra)](https://github.com/NVIDIA/cutlass)

### Papers & Talks
- "Anatomy of High-Performance GEMM" (Goto & van de Geijn)
- [How to Optimize GEMM on GPU (NVIDIA Blog)](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)
- "Radix Sort Revisited" (Satish et al.)

### Reading
- "Programming Massively Parallel Processors" - Chapter 11 (Libraries)

## Practice Exercises

1. **GEMM variants**: Rectangular matrices, batched GEMM
2. **Transpose variants**: Non-square matrices, 3D arrays
3. **Custom sorting**: Sort by key-value pairs
4. **Thrust algorithms**: Use thrust for data processing pipeline
5. **Library comparison**: Benchmark cuBLAS vs custom GEMM

## Debugging Tips

### Profile GEMM:
```bash
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed program
# Target: 60-80% for optimized GEMM
```

### Check transpose bandwidth:
```bash
nvprof --metrics dram_read_throughput,dram_write_throughput program
# Should be close to theoretical bandwidth
```

### Verify sorting correctness:
```cuda
// Check if output is sorted
for (int i = 1; i < n; i++) {
    assert(data[i-1] <= data[i]);
}
```

## Next Phase

Once comfortable with Phase 5, move to:
**Phase 6: Streams & Concurrency** - Learn asynchronous execution, overlap computation with transfers, and multi-GPU programming.

**Path**: `../phase6/README.md`

---

**Pro Tip**: GEMM optimization is the ultimate test of CUDA skills. If you can write efficient GEMM, you understand all core optimization techniques!

## Questions to Test Your Understanding

1. What is register blocking and why does it help GEMM?
2. Why does cuBLAS use column-major layout?
3. How do you eliminate bank conflicts in transpose?
4. What's the complexity of bitonic vs radix sort?
5. When should you use Thrust vs custom kernels?
6. How many passes does radix sort need for 32-bit integers?
7. What performance percentage of cuBLAS is achievable with custom GEMM?
8. How does prefix sum enable radix sort?

If you can implement optimized GEMM and understand all algorithms in this phase, you're ready for Phase 6!
