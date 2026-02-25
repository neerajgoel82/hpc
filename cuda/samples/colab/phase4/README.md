# Phase 4: Advanced Memory (Weeks 7-8)

## Overview
Phase 4 explores specialized memory types and advanced synchronization primitives. You'll master texture memory, constant memory, zero-copy techniques, and cooperative groups for flexible thread coordination.

## Notebooks in This Phase

### 18_texture_memory.ipynb
**Duration**: 2 hours
**Learning Objectives**:
- Use CUDA texture memory for cached reads
- Understand texture hardware benefits
- Implement 2D texture access
- Apply texture memory to real problems

**Key Concepts**:
- Texture cache (separate from L1/L2)
- 2D spatial locality optimization
- Texture objects vs texture references
- Normalized vs unnormalized coordinates
- Linear interpolation hardware support

**Applications**:
- Image processing with texture memory
- Nearest-neighbor interpolation
- Matrix operations with cached access

---

### 19_constant_memory.ipynb
**Duration**: 1.5 hours
**Learning Objectives**:
- Use `__constant__` memory
- Understand broadcast optimization
- Learn when constant memory helps
- Measure constant memory performance

**Key Concepts**:
- Constant memory cache (64KB)
- Broadcast to all threads in a warp
- Read-only access
- Best when all threads read same address
- Serialization when threads diverge

**Implementations**:
- Convolution with constant kernel
- Matrix operations with constant parameters
- Performance comparison: constant vs global

---

### 20_zero_copy_memory.ipynb
**Duration**: 1.5 hours
**Learning Objectives**:
- Use zero-copy (mapped) memory
- Access host memory from device
- Understand PCIe overhead
- Learn when zero-copy is beneficial

**Key Concepts**:
- `cudaHostAlloc` with mapped flags
- Direct device access to host memory
- No explicit transfers needed
- Trade-offs: convenience vs performance
- Best for data accessed once

**Use Cases**:
- Small data reads
- Irregular access patterns
- Avoiding explicit transfers

---

### 21_atomic_operations_advanced.ipynb
**Duration**: 2 hours
**Learning Objectives**:
- Master all atomic functions
- Implement lock-free data structures
- Use `atomicCAS` for complex operations
- Optimize atomic performance

**Key Concepts**:
- Full atomic API: Add, Sub, Min, Max, Inc, Dec, CAS, Exch
- Compare-and-swap (CAS) for custom operations
- Atomic scope: global, shared, system
- Warp aggregation techniques
- Lock-free algorithms

**Implementations**:
- Atomic max/min reduction
- Lock-free linked list insertion
- Atomic floating-point operations
- Optimized histogram with privatization

---

### 22_cooperative_groups.ipynb ⭐ KEY NOTEBOOK
**Duration**: 2-3 hours
**Learning Objectives**:
- Use Cooperative Groups API
- Create flexible thread groups
- Synchronize arbitrary thread sets
- Implement advanced parallel patterns

**Key Concepts**:
- Thread groups: grid, block, tile, partition
- `group.sync()` for flexible synchronization
- Coalesced groups (active threads only)
- Tiled partitions (e.g., warp subsets)
- Grid-wide synchronization

**Implementations**:
- Reduction with tiled groups
- Warp-level operations with partitions
- Multi-block algorithms with grid groups

---

### 23_multi_kernel_synchronization.ipynb
**Duration**: 1.5 hours
**Learning Objectives**:
- Synchronize between kernel launches
- Use CUDA events for dependencies
- Implement multi-stage algorithms
- Understand kernel launch overhead

**Key Concepts**:
- Implicit synchronization (kernel boundaries)
- CUDA events for timing and sync
- Device-side synchronization (Phase 9)
- Multi-pass algorithms
- Pipeline patterns

**Implementations**:
- Two-pass reduction
- Multi-stage sorting
- Producer-consumer patterns

---

## Learning Path

```
18-texture-memory
        ↓
19-constant-memory
        ↓
20-zero-copy-memory
        ↓
21-atomics-advanced
        ↓
22-cooperative-groups ⭐
        ↓
23-multi-kernel-sync
```

## Prerequisites
- Completed Phases 1-3
- Strong memory management skills
- Understanding of synchronization
- Atomic operations basics

## Success Criteria

By the end of Phase 4, you should be able to:
- [ ] Use texture memory for optimized access
- [ ] Apply constant memory appropriately
- [ ] Implement zero-copy strategies
- [ ] Master advanced atomic operations
- [ ] Use Cooperative Groups API effectively
- [ ] Synchronize across multiple kernels
- [ ] Choose the right memory type for each use case
- [ ] Implement lock-free algorithms

## Memory Types Summary

| Memory Type | Size | Scope | Speed | Use Case |
|-------------|------|-------|-------|----------|
| **Registers** | ~64KB/SM | Thread | Fastest | Automatic variables |
| **Shared** | 48-96KB/SM | Block | Very Fast | Inter-thread data sharing |
| **Constant** | 64KB | Grid | Fast (cached) | Read-only data, broadcast |
| **Texture** | - | Grid | Fast (cached) | 2D spatial locality |
| **Global** | GB | Grid | Slow | Large data, coalesced access |
| **Local** | - | Thread | Slow | Register spills |

## Choosing the Right Memory

```
Decision Tree:
├─ Data read-only?
│  ├─ YES → All threads read same value?
│  │  ├─ YES → Use constant memory
│  │  └─ NO → Has spatial locality?
│  │     ├─ YES → Use texture memory
│  │     └─ NO → Use global memory
│  └─ NO → Shared within block?
│     ├─ YES → Use shared memory
│     └─ NO → Use global memory
```

## Performance Expectations

- **Texture memory**: 2-3x faster for 2D access patterns
- **Constant memory**: 10-100x faster for broadcast reads
- **Zero-copy**: Slower than device memory, but convenient
- **Cooperative Groups**: Similar performance, more flexible
- **Optimized atomics**: 5-10x faster with privatization

## Common Pitfalls

1. **Texture Memory Misuse**
   ```cuda
   // Only helps with spatial locality
   // Don't use for linear sequential access - use global memory
   ```

2. **Constant Memory Divergence**
   ```cuda
   // BAD - threads read different constant addresses
   __constant__ float data[1000];
   float val = data[threadIdx.x];  // Serialized! Slow!
   
   // GOOD - all threads read same address
   float val = data[0];  // Broadcast! Fast!
   ```

3. **Zero-Copy Performance**
   ```cuda
   // Zero-copy is SLOW for repeated access
   // Only use for data accessed once
   ```

4. **AtomicCAS Incorrect Usage**
   ```cuda
   // Must use loop for atomicCAS
   int old = atomicCAS(&addr, compare, val);
   while (old != compare) {
       compare = old;
       old = atomicCAS(&addr, compare, val);
   }
   ```

## Advanced Patterns

### 1. Lock-Free Stack Push
```cuda
__device__ void push(int* stack, int* top, int value) {
    int old_top = *top;
    int new_top;
    do {
        new_top = old_top + 1;
        old_top = atomicCAS(top, old_top, new_top);
    } while (old_top != new_top - 1);
    stack[new_top] = value;
}
```

### 2. Warp Aggregated Atomics
```cuda
// Reduce atomics by aggregating within warp first
int mask = __ballot_sync(0xffffffff, predicate);
int leader = __ffs(mask) - 1;
int warp_sum = 0;
if (predicate) {
    warp_sum = /* warp reduction */;
}
if (threadIdx.x % 32 == leader) {
    atomicAdd(&global_sum, warp_sum);
}
```

### 3. Cooperative Groups Reduction
```cuda
#include <cooperative_groups.h>
using namespace cooperative_groups;

__device__ int reduce(thread_block_tile<32> tile, int val) {
    for (int offset = tile.size()/2; offset > 0; offset /= 2) {
        val += tile.shfl_down(val, offset);
    }
    return val;
}
```

## Time Estimate
- **Fast pace**: 1.5 weeks (3-4 hours/day)
- **Moderate pace**: 2 weeks (2 hours/day)
- **Relaxed pace**: 3 weeks (1-1.5 hours/day)

## Additional Resources

### NVIDIA Documentation
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) - Chapter 3.2 (Memory Types)
- [Cooperative Groups Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups)

### Reading
- "Programming Massively Parallel Processors" - Chapter 6 (Memory)
- [Using Texture Memory (NVIDIA Blog)](https://developer.nvidia.com/blog/cuda-pro-tip-kepler-texture-objects-improve-performance-and-flexibility/)

## Practice Exercises

1. **2D Gaussian blur** with texture memory
2. **Sobel edge detection** with constant kernel
3. **Lock-free queue** with atomicCAS
4. **Bitonic sort** with cooperative groups
5. **Custom atomic operations** (e.g., atomic max for floats)

## Debugging Tips

### Profile texture cache hit rate:
```bash
nvprof --metrics tex_cache_hit_rate program
```

### Check constant memory broadcasts:
```bash
nvprof --metrics const_cache_hit_rate program
```

### Measure atomic throughput:
```bash
nvprof --metrics atomic_transactions_per_request program
```

## Next Phase

Once comfortable with Phase 4, move to:
**Phase 5: Advanced Algorithms** - Implement production-quality matrix operations, sorting algorithms, and use cuBLAS/Thrust libraries.

**Path**: `../phase5/README.md`

---

**Pro Tip**: Texture and constant memory are often overlooked but can provide significant speedups for the right access patterns. Profile first, then optimize!

## Questions to Test Your Understanding

1. When should you use texture memory vs global memory?
2. Why is constant memory fast for broadcast reads?
3. What are the trade-offs of zero-copy memory?
4. How does atomicCAS work and when would you use it?
5. What are Cooperative Groups and how do they differ from traditional sync?
6. How do you synchronize between kernel launches?
7. How can you optimize atomic operations?
8. What is warp aggregation?

If you understand all memory types and can use Cooperative Groups effectively, you're ready for Phase 5!
