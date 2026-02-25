# Phase 3: Optimization (Weeks 5-6)

## Overview
Phase 3 teaches you how to write efficient parallel algorithms. You'll learn warp-level programming, occupancy optimization, and implement fundamental parallel patterns like reduction and prefix sum.

## Notebooks in This Phase

### 12_warp_divergence_and_branch_efficiency.ipynb
**Duration**: 1.5 hours
**Learning Objectives**:
- Understand warp execution model
- Identify and minimize branch divergence
- Measure performance impact of divergence
- Write branch-efficient code

**Key Concepts**:
- Warps (32 threads execute in lockstep)
- SIMT (Single Instruction Multiple Thread)
- Branch divergence cost
- Predication vs branching

**Practice Problems**:
- Rewrite branching code to reduce divergence
- Measure divergence with profiling metrics
- Optimize conditional execution

---

### 13_warp_shuffle_operations.ipynb
**Duration**: 2 hours
**Learning Objectives**:
- Use warp shuffle intrinsics
- Implement warp-level reductions
- Share data within a warp without shared memory
- Master `__shfl_*` functions

**Key Concepts**:
- `__shfl_sync()`, `__shfl_down_sync()`, `__shfl_up_sync()`
- `__shfl_xor_sync()` for butterfly patterns
- Warp-level primitives
- Synchronization masks

**Implementations**:
- Warp-level reduction
- Warp-level scan (prefix sum)
- Efficient data exchange within warps

---

### 14_occupancy_tuning_and_resource_optimization.ipynb
**Duration**: 2 hours
**Learning Objectives**:
- Understand GPU occupancy
- Balance threads, registers, and shared memory
- Use CUDA Occupancy Calculator
- Optimize launch configurations

**Key Concepts**:
- Occupancy: active warps / maximum warps
- Register pressure
- Shared memory limitations
- Thread block size selection
- Occupancy vs performance trade-offs

**Tools**:
- `nvcc --ptxas-options=-v` for resource usage
- CUDA Occupancy Calculator API
- `__launch_bounds__` directive

---

### 15_parallel_reduction.ipynb ⭐ KEY NOTEBOOK
**Duration**: 2-3 hours
**Learning Objectives**:
- Implement parallel reduction (sum, min, max)
- Master tree-based algorithms
- Combine shared memory + warp shuffles
- Achieve optimal performance

**Key Concepts**:
- Reduction tree patterns
- Sequential addressing
- Shared memory reduction
- Warp-level final reduction
- Multiple-pass strategies

**Implementations**:
- Naive global memory reduction
- Shared memory tree reduction
- Optimized with warp shuffles
- Multi-block reduction strategies

---

### 16_prefix_sum_scan.ipynb
**Duration**: 2-3 hours
**Learning Objectives**:
- Implement parallel prefix sum (scan)
- Understand inclusive vs exclusive scan
- Learn Blelloch algorithm
- Apply scan to other problems

**Key Concepts**:
- Inclusive scan vs exclusive scan
- Hillis-Steele algorithm (work-inefficient)
- Blelloch algorithm (work-efficient)
- Bank-conflict-free access patterns
- Multi-block scan strategies

**Implementations**:
- Simple scan with shared memory
- Optimized scan with no bank conflicts
- Applications: stream compaction, radix sort

---

### 17_histogram_and_atomic_operations.ipynb
**Duration**: 1.5 hours
**Learning Objectives**:
- Use atomic operations correctly
- Implement parallel histogram
- Understand atomic performance costs
- Learn atomic optimization strategies

**Key Concepts**:
- `atomicAdd`, `atomicMin`, `atomicMax`, `atomicCAS`
- Atomic operation overhead
- Shared memory atomics (faster)
- Privatization techniques
- Warp aggregation

**Implementations**:
- Naive global memory histogram
- Optimized with shared memory atomics
- Privatization and reduction strategy

---

## Learning Path

```
12-warp-divergence
        ↓
13-warp-shuffle
        ↓
14-occupancy-tuning
        ↓
15-parallel-reduction ⭐
        ↓
16-prefix-sum-scan ⭐
        ↓
17-histogram-atomics
```

## Prerequisites
- Completed Phases 1-2
- Strong understanding of shared memory
- Comfortable with thread synchronization
- Basic algorithm design knowledge

## Success Criteria

By the end of Phase 3, you should be able to:
- [ ] Identify and minimize warp divergence
- [ ] Use warp shuffle intrinsics effectively
- [ ] Calculate and optimize GPU occupancy
- [ ] Implement parallel reduction from scratch
- [ ] Implement parallel prefix sum (scan)
- [ ] Use atomic operations safely and efficiently
- [ ] Choose appropriate parallel patterns for problems
- [ ] Profile and optimize CUDA kernels

## Key Algorithms & Patterns

### 1. Parallel Reduction
**Use cases**: Sum, min, max, dot product, norm calculations

```cuda
// Tree-based reduction in shared memory
__shared__ float sdata[256];
sdata[tid] = data[idx];
__syncthreads();

for (int s = blockDim.x/2; s > 0; s >>= 1) {
    if (tid < s) {
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
}
```

### 2. Prefix Sum (Scan)
**Use cases**: Stream compaction, radix sort, load balancing

```cuda
// Hillis-Steele scan (simple, work-inefficient)
for (int offset = 1; offset < n; offset *= 2) {
    if (tid >= offset) {
        temp = sdata[tid] + sdata[tid - offset];
    }
    __syncthreads();
    sdata[tid] = temp;
    __syncthreads();
}
```

### 3. Warp-Level Reduction
**Use cases**: Fast local reductions without shared memory

```cuda
// Reduce within a warp using shuffles
for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
}
```

## Performance Expectations

After Phase 3 optimizations:

- **Reduction**: 100-200 GB/s bandwidth utilization
- **Scan**: 50-100 GB/s throughput
- **Histogram**: 10-50x faster than CPU
- **Warp shuffles**: 2-3x faster than shared memory for small data

## Common Pitfalls

1. **Warp Divergence**
   ```cuda
   // BAD - divergent branches
   if (tid % 2 == 0) {
       // Even threads
   } else {
       // Odd threads - causes divergence!
   }
   
   // BETTER - less divergence
   if (tid < n/2) {
       // First half of threads
   }
   ```

2. **Incorrect Shuffle Masks**
   ```cuda
   // IMPORTANT: Use correct active mask (all 1s = 0xffffffff)
   val = __shfl_down_sync(0xffffffff, val, offset);
   ```

3. **Occupancy Obsession**
   ```
   Higher occupancy ≠ Always better performance
   Focus on memory throughput, not just occupancy!
   ```

4. **Atomic Contention**
   ```cuda
   // BAD - high contention on global atomics
   atomicAdd(&global_counter, 1);
   
   // BETTER - use shared memory first
   __shared__ int local_counter;
   atomicAdd(&local_counter, 1);
   __syncthreads();
   if (tid == 0) atomicAdd(&global_counter, local_counter);
   ```

## Optimization Checklist

When optimizing kernels:

1. ✅ **Profile first** - Identify bottlenecks
2. ✅ **Memory bandwidth** - Usually the #1 bottleneck
3. ✅ **Minimize divergence** - Keep warps together
4. ✅ **Use warp shuffles** - Faster than shared memory for small data
5. ✅ **Check occupancy** - But prioritize bandwidth
6. ✅ **Reduce atomics** - Use privatization techniques
7. ✅ **Minimize synchronization** - Each `__syncthreads()` has cost

## Time Estimate
- **Fast pace**: 1.5 weeks (3-4 hours/day)
- **Moderate pace**: 2 weeks (2 hours/day)
- **Relaxed pace**: 3 weeks (1-1.5 hours/day)

## Additional Resources

### NVIDIA Documentation
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) - Chapter 5.4 (Warp Shuffle)
- [CUDA C Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) - Optimization Strategies

### Algorithms
- [Parallel Prefix Sum (Scan) in CUDA (Harris et al.)](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)
- [Optimizing Parallel Reduction in CUDA (Harris)](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)

### Reading
- "Programming Massively Parallel Processors" - Chapters 8-10
- "CUDA by Example" - Chapter 9

## Practice Exercises

1. **Parallel max/min** reduction
2. **Vector dot product** using reduction
3. **Exclusive scan** implementation
4. **Stream compaction** using scan
5. **Sparse matrix operations** with atomics
6. **Custom reduction operator** (e.g., finding mode)

## Debugging Tools

### Check for divergence:
```bash
nvprof --metrics branch_efficiency program
# Higher is better (100% = no divergence)
```

### Check occupancy:
```bash
nvprof --metrics achieved_occupancy program
# Typical target: 50-75%
```

### Profile atomic operations:
```bash
nvprof --metrics atomic_transactions program
# Lower is better (less contention)
```

## Next Phase

Once comfortable with Phase 3, move to:
**Phase 4: Advanced Memory** - Learn about texture memory, constant memory, zero-copy, and advanced synchronization.

**Path**: `../phase4/README.md`

---

**Pro Tip**: Reduction and scan are fundamental building blocks. Master them well - you'll use these patterns in countless CUDA programs!

## Questions to Test Your Understanding

1. What is a warp and why does divergence matter?
2. How do warp shuffle functions work?
3. What is GPU occupancy and how do you calculate it?
4. Why isn't 100% occupancy always optimal?
5. Explain the tree-based reduction algorithm.
6. What's the difference between inclusive and exclusive scan?
7. When should you use atomics vs other approaches?
8. How do you reduce atomic contention?

If you can implement reduction and scan from scratch and explain warp-level programming, you're ready for Phase 4!
