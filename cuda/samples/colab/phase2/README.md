# Phase 2: Memory Management (Weeks 3-4)

## Overview
Phase 2 focuses on CUDA memory hierarchy and optimization techniques. You'll learn about different memory types, understand memory bandwidth, and master shared memory for significant performance improvements.

## Notebooks in This Phase

### 06_memory_basics_and_data_transfer.ipynb
**Duration**: 1 hour
**Learning Objectives**:
- Master `cudaMalloc`, `cudaMemcpy`, `cudaFree`
- Understand host-device memory transfer patterns
- Learn about pinned (page-locked) memory
- Optimize data transfer strategies

**Key Concepts**:
- Device memory allocation
- Memory copy directions (H2D, D2H, D2D)
- Pinned vs pageable memory
- Asynchronous memory transfers

---

### 07_memory_bandwidth_benchmarking.ipynb
**Duration**: 1.5 hours
**Learning Objectives**:
- Measure memory bandwidth
- Understand PCIe transfer bottlenecks
- Benchmark different memory operations
- Learn effective bandwidth vs theoretical bandwidth

**Key Concepts**:
- Memory bandwidth calculation
- PCIe bus limitations
- CUDA events for precise timing
- Performance metrics

**Practice Problems**:
- Measure bandwidth for different transfer sizes
- Compare pinned vs pageable memory performance
- Optimize transfer patterns

---

### 08_unified_memory_and_managed_memory.ipynb
**Duration**: 1.5 hours
**Learning Objectives**:
- Use unified memory (`cudaMallocManaged`)
- Understand automatic memory migration
- Learn about memory prefetching
- When to use unified vs explicit memory

**Key Concepts**:
- Unified Memory model
- Automatic page migration
- `cudaMemPrefetchAsync`
- `cudaMemAdvise` hints
- Performance trade-offs

**Implementations**:
- Unified memory vector operations
- Comparison with explicit memory management
- Prefetching demonstrations

---

### 09_shared_memory_basics.ipynb
**Duration**: 2 hours
**Learning Objectives**:
- Understand on-chip shared memory
- Declare and use `__shared__` variables
- Implement shared memory reductions
- Master thread synchronization with `__syncthreads()`

**Key Concepts**:
- Shared memory vs global memory
- Bank conflicts (introduction)
- Thread synchronization
- Shared memory size limits
- Dynamic shared memory

**Implementations**:
- Shared memory vector operations
- Parallel reduction with shared memory
- Performance comparison: global vs shared

---

### 10_tiled_matrix_multiplication.ipynb ⭐ KEY NOTEBOOK
**Duration**: 2-3 hours
**Learning Objectives**:
- Implement tiled matrix multiplication
- Use shared memory for optimization
- Understand memory reuse patterns
- Achieve 10-20x speedup over naive version

**Key Concepts**:
- Tiling strategy
- Shared memory tiles
- Memory access pattern optimization
- Algorithm complexity analysis

**Implementations**:
- Naive matrix multiply (global memory only)
- Tiled matrix multiply (shared memory)
- Performance comparison and analysis

---

### 11_memory_coalescing_demonstration.ipynb
**Duration**: 1.5 hours
**Learning Objectives**:
- Understand coalesced memory access
- Identify and fix uncoalesced access patterns
- Measure impact on performance
- Learn stride patterns

**Key Concepts**:
- Coalesced vs strided access
- Memory transaction efficiency
- Warp-level memory access
- Cache line utilization

**Practice Problems**:
- Transpose matrix with coalescing
- Fix uncoalesced access patterns
- Measure bandwidth improvements

---

## Learning Path

```
06-memory-basics
        ↓
07-bandwidth-benchmarking
        ↓
08-unified-memory
        ↓
09-shared-memory-basics
        ↓
10-tiled-matrix-multiply ⭐
        ↓
11-memory-coalescing
```

## Prerequisites
- Completed Phase 1
- Comfortable with CUDA kernel basics
- Understanding of matrix operations
- Basic performance analysis skills

## Success Criteria

By the end of Phase 2, you should be able to:
- [ ] Allocate and manage device memory efficiently
- [ ] Measure and optimize memory bandwidth
- [ ] Use unified memory appropriately
- [ ] Implement shared memory optimizations
- [ ] Write tiled matrix multiplication from scratch
- [ ] Identify and fix uncoalesced memory access
- [ ] Explain the CUDA memory hierarchy
- [ ] Choose the right memory type for your algorithm

## Key Takeaways

### Memory Hierarchy (Fast → Slow)
1. **Registers** (per-thread, fastest)
2. **Shared Memory** (per-block, on-chip, ~100x faster than global)
3. **L1/L2 Cache** (automatic, transparent)
4. **Global Memory** (device DRAM, largest, slowest)
5. **Host Memory** (system RAM, PCIe transfer overhead)

### Memory Optimization Strategies
1. **Minimize global memory access** - Use shared memory for reuse
2. **Coalesce memory access** - Adjacent threads access adjacent memory
3. **Avoid bank conflicts** - Access different shared memory banks
4. **Use pinned memory** - For faster host-device transfers
5. **Overlap transfers** - Hide latency with computation (Phase 6)

## Common Pitfalls

1. **Forgetting __syncthreads() with shared memory**
   ```cuda
   __shared__ float tile[TILE_SIZE];
   tile[threadIdx.x] = data[idx];
   __syncthreads();  // Critical! Wait for all threads
   // Now safe to read tile[]
   ```

2. **Shared memory size limits**
   ```cuda
   // Check shared memory limit (typically 48KB per block)
   cudaDeviceProp prop;
   cudaGetDeviceProperties(&prop, 0);
   printf("Shared mem per block: %zu\n", prop.sharedMemPerBlock);
   ```

3. **Bank conflicts**
   ```cuda
   // BAD - all threads access same bank
   __shared__ float arr[32];
   float val = arr[threadIdx.x % 32];  // Conflict!
   
   // GOOD - sequential access
   float val = arr[threadIdx.x];  // No conflict
   ```

4. **Uncoalesced access**
   ```cuda
   // BAD - strided access
   data[threadIdx.x * stride];  // Uncoalesced
   
   // GOOD - sequential access
   data[threadIdx.x];  // Coalesced
   ```

## Performance Expectations

After completing Phase 2, typical speedups vs naive implementations:

- **Shared memory reduction**: 5-10x faster than global memory
- **Tiled matrix multiply**: 10-20x faster than naive version
- **Coalesced access**: 5-10x faster than strided access
- **Pinned memory transfers**: 2-3x faster than pageable memory

## Time Estimate
- **Fast pace**: 1 week (3-4 hours/day)
- **Moderate pace**: 2 weeks (1-2 hours/day)
- **Relaxed pace**: 3 weeks (1 hour/day)

## Additional Resources

### NVIDIA Documentation
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) - Chapter 5 (Memory)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) - Memory Optimization

### Reading
- "CUDA by Example" - Chapter 5-6
- "Programming Massively Parallel Processors" - Chapter 4-5

### Tools
- Nsight Compute for memory profiling (Phase 7)
- `nvprof --metrics` for bandwidth analysis

## Practice Exercises

1. **Vector dot product** with shared memory reduction
2. **Matrix transpose** with shared memory and coalescing
3. **1D convolution** using shared memory tiling
4. **Histogram** using shared memory atomics
5. **Prefix sum** (scan) using shared memory

## Debugging Tips

### Check shared memory usage:
```cuda
nvcc --ptxas-options=-v program.cu
// Look for: "bytes shared memory" per block
```

### Detect bank conflicts:
```bash
nvprof --metrics shared_load_transactions_per_request program
// Value > 1.0 indicates conflicts
```

### Measure memory bandwidth:
```cuda
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start);
// ... kernel ...
cudaEventRecord(stop);
cudaEventSynchronize(stop);
float ms;
cudaEventElapsedTime(&ms, start, stop);
float bandwidth = (bytes / ms) / 1e6;  // GB/s
```

## Next Phase

Once comfortable with Phase 2, move to:
**Phase 3: Optimization** - Learn about warp-level operations, occupancy tuning, and advanced parallel algorithms.

**Path**: `../phase3/README.md`

---

**Pro Tip**: Memory optimization is the #1 performance factor in CUDA. Master shared memory and coalescing - they're used in almost every high-performance CUDA program!

## Questions to Test Your Understanding

1. What is the difference between global and shared memory?
2. How much shared memory is available per block on modern GPUs?
3. What is memory coalescing and why does it matter?
4. When should you use unified memory vs explicit memory management?
5. What is `__syncthreads()` and when is it required?
6. How do you calculate memory bandwidth from transfer time?
7. What is a bank conflict and how do you avoid it?
8. Why is pinned memory faster for transfers?

If you can answer these confidently and implement tiled matrix multiplication, you're ready for Phase 3!
