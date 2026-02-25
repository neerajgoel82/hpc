# Phase 7: Performance Engineering (Weeks 12-13)

## Overview
Phase 7 focuses on professional-grade optimization and debugging. You'll learn to use NVIDIA's profiling tools, debug CUDA programs effectively, optimize with advanced techniques, and apply best practices.

## Notebooks in This Phase

### 36_profiling_with_nsight.ipynb ⭐ KEY NOTEBOOK
**Duration**: 3 hours
**Learning Objectives**:
- Use Nsight Compute for kernel profiling
- Use Nsight Systems for system-level profiling
- Identify performance bottlenecks
- Interpret profiling metrics
- Apply optimization based on profile data

**Key Concepts**:
- Nsight Compute: Kernel-level profiler
- Nsight Systems: System-wide timeline profiler
- Key metrics: bandwidth, occupancy, throughput
- Roofline model
- Bottleneck analysis

**Tools**:
```bash
# Nsight Compute (kernel profiler)
ncu --set full -o report ./program

# Nsight Systems (timeline profiler)
nsys profile -o timeline ./program
```

---

### 37_debugging_cuda_programs.ipynb
**Duration**: 2 hours
**Learning Objectives**:
- Debug CUDA errors systematically
- Use cuda-gdb for debugging
- Use Compute Sanitizer for memory errors
- Handle race conditions
- Validate numerical correctness

**Key Concepts**:
- Error checking patterns
- `cuda-gdb` debugger
- `compute-sanitizer` for memory errors
- Race detection tools
- Debugging device code

**Common Errors**:
- Kernel launch failures
- Memory access violations
- Race conditions
- Uninitialized memory
- Synchronization bugs

---

### 38_kernel_fusion.ipynb
**Duration**: 2 hours
**Learning Objectives**:
- Fuse multiple kernels into one
- Reduce kernel launch overhead
- Eliminate intermediate transfers
- Optimize memory bandwidth

**Key Concepts**:
- Kernel launch overhead (~5-10 μs)
- Memory bandwidth savings
- Fused kernel complexity
- Trade-offs: fusion vs modularity

**Example**:
```cuda
// Before: 3 kernels, 3 launches, 3 memory passes
kernel1<<<grid, block>>>(d_in, d_temp1);
kernel2<<<grid, block>>>(d_temp1, d_temp2);
kernel3<<<grid, block>>>(d_temp2, d_out);

// After: 1 fused kernel, 1 launch, 1 memory pass
fused_kernel<<<grid, block>>>(d_in, d_out);
// 3x faster due to bandwidth reduction
```

---

### 39_fast_math_and_intrinsics.ipynb
**Duration**: 1.5 hours
**Learning Objectives**:
- Use CUDA math intrinsics
- Enable fast math optimizations
- Understand accuracy trade-offs
- Optimize mathematical operations

**Key Concepts**:
- `--use_fast_math` compiler flag
- Intrinsic functions: `__fdividef`, `__sinf`, `__cosf`
- Single vs double precision
- Accuracy vs performance trade-offs
- Special function units (SFUs)

**Performance**:
- Fast math: 2-10x faster for math-heavy code
- Lower precision: 2x faster (FP32 vs FP64)

---

### 40_advanced_optimization_best_practices.ipynb
**Duration**: 2 hours
**Learning Objectives**:
- Apply all optimization techniques
- Understand optimization priorities
- Use compiler optimizations
- Follow CUDA best practices

**Key Concepts**:
- Optimization priority ranking
- Amdahl's Law application
- Compiler flags and options
- Code generation tuning
- Architecture-specific optimizations

**Best Practices Checklist**:
- Memory optimization (highest priority)
- Instruction optimization
- Control flow optimization
- API optimization

---

## Learning Path

```
36-profiling-nsight ⭐
        ↓
37-debugging-cuda
        ↓
38-kernel-fusion
        ↓
39-fast-math
        ↓
40-best-practices
```

## Prerequisites
- Completed Phases 1-6
- Experience with all optimization techniques
- Understanding of performance metrics
- Comfortable with command-line tools

## Success Criteria

By the end of Phase 7, you should be able to:
- [ ] Profile CUDA applications with Nsight tools
- [ ] Identify and fix performance bottlenecks
- [ ] Debug CUDA programs systematically
- [ ] Apply kernel fusion when appropriate
- [ ] Use fast math optimizations effectively
- [ ] Follow all CUDA best practices
- [ ] Optimize code to near-theoretical performance
- [ ] Make informed optimization trade-offs

## Profiling Metrics

### Key Performance Metrics:

1. **Memory Metrics**:
   - DRAM throughput (GB/s)
   - Memory bandwidth utilization (%)
   - Cache hit rates
   - Coalescing efficiency

2. **Compute Metrics**:
   - Achieved occupancy (%)
   - SM utilization (%)
   - Warp execution efficiency
   - IPC (instructions per cycle)

3. **Instruction Metrics**:
   - Branch efficiency
   - Divergence percentage
   - FP32/FP64 throughput
   - Special function usage

### Performance Targets:

| Metric | Target | Notes |
|--------|--------|-------|
| **Memory bandwidth** | 80-95% | Bandwidth-bound kernels |
| **Occupancy** | 50-75% | Not always need 100% |
| **SM utilization** | 90-100% | GPU should be busy |
| **Branch efficiency** | 95-100% | Minimize divergence |
| **Coalescing** | 90-100% | Good memory patterns |

## Optimization Priority (Amdahl's Law)

```
Priority 1: Memory Bandwidth (80% of problems)
  ├─ Coalesced access
  ├─ Shared memory usage
  └─ Minimize DRAM transactions

Priority 2: Parallelism (15% of problems)
  ├─ Sufficient work per thread
  ├─ Good occupancy
  └─ Minimize synchronization

Priority 3: Instructions (5% of problems)
  ├─ Fast math
  ├─ Minimize divergence
  └─ Use intrinsics
```

**Rule**: Profile first! Don't optimize blindly.

## Common Performance Issues

### 1. Uncoalesced Memory Access
```bash
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
# High value indicates poor coalescing
```

### 2. Low Occupancy
```bash
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active
# Target: 50-75% for most kernels
```

### 3. Divergence
```bash
ncu --metrics smsp__sass_average_branch_targets_threads_uniform.pct
# Target: >90%
```

### 4. Bank Conflicts
```bash
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
# Should be near zero
```

## Debugging Workflow

### 1. Error Checking
```cuda
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

CUDA_CHECK(cudaMalloc(&d_ptr, size));
```

### 2. Memory Errors
```bash
compute-sanitizer --tool memcheck ./program
# Detects: out-of-bounds, uninitialized memory, leaks
```

### 3. Race Conditions
```bash
compute-sanitizer --tool racecheck ./program
# Detects: data races, synchronization issues
```

### 4. Interactive Debugging
```bash
cuda-gdb ./program
(cuda-gdb) break kernel_name
(cuda-gdb) run
(cuda-gdb) cuda thread
(cuda-gdb) print variable
```

## Advanced Optimization Techniques

### 1. Instruction-Level Parallelism (ILP)
```cuda
// Unroll to keep pipeline full
#pragma unroll
for (int i = 0; i < 8; i++) {
    result += data[i] * coef[i];
}
```

### 2. Vectorized Loads
```cuda
// Load 4 floats at once (16-byte aligned)
float4 val = *reinterpret_cast<float4*>(&data[idx]);
```

### 3. Register Blocking
```cuda
// Compute multiple outputs per thread
float results[4];
for (int i = 0; i < 4; i++) {
    results[i] = compute(i);
}
```

### 4. Warp Specialization
```cuda
// Different warps do different work
if (warpId < N_WARPS_COMPUTE) {
    // Compute-intensive work
} else {
    // Memory-intensive work
}
```

## Compiler Optimization Flags

```bash
# Standard optimizations
nvcc -O3 program.cu -o program

# Fast math (less accurate, much faster)
nvcc -use_fast_math program.cu -o program

# Architecture-specific
nvcc -arch=sm_80 program.cu -o program

# Show resource usage
nvcc --ptxas-options=-v program.cu -o program

# Generate PTX for inspection
nvcc --ptx program.cu

# Aggressive optimization
nvcc -O3 -use_fast_math -arch=sm_80 --maxrregcount=64 program.cu
```

## Time Estimate
- **Fast pace**: 1.5 weeks (3-4 hours/day)
- **Moderate pace**: 2 weeks (2 hours/day)
- **Relaxed pace**: 2-3 weeks (1-2 hours/day)

## Additional Resources

### NVIDIA Documentation
- [CUDA C Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)

### Profiling Guides
- [Profiling Guide](https://docs.nvidia.com/cuda/profiler-users-guide/)
- [Kernel Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)

### Videos
- "CUDA Performance Optimization" (GTC talks)
- "Nsight Tools Tutorial" (NVIDIA)

## Practice Exercises

1. **Profile existing code**: Use Nsight to find bottlenecks
2. **Fix memory issues**: Use compute-sanitizer to find and fix bugs
3. **Kernel fusion exercise**: Fuse 3 separate kernels
4. **Fast math comparison**: Measure accuracy vs performance trade-offs
5. **Optimization challenge**: Optimize a slow kernel to near-peak

## Professional Development Tips

### Code Review Checklist:
- [ ] All CUDA calls checked for errors
- [ ] Memory properly allocated and freed
- [ ] Coalesced memory access
- [ ] Appropriate use of shared memory
- [ ] Minimal warp divergence
- [ ] Good occupancy (not necessarily 100%)
- [ ] Profiled and optimized
- [ ] Documented optimization choices

## Next Phase

Once comfortable with Phase 7, move to:
**Phase 8: Real Applications** - Build complete applications using CUDA libraries and techniques learned.

**Path**: `../phase8/README.md`

---

**Pro Tip**: "Premature optimization is the root of all evil" - Profile first, then optimize the bottleneck. Don't guess!

## Questions to Test Your Understanding

1. What's the difference between Nsight Compute and Nsight Systems?
2. What metrics indicate memory bandwidth bottleneck?
3. How do you debug kernel launch failures?
4. When should you fuse kernels?
5. What are the trade-offs of `--use_fast_math`?
6. What is a good occupancy target and why?
7. How do you detect race conditions?
8. What is the optimization priority ranking?

If you can profile, debug, and systematically optimize CUDA code, you're ready for Phase 8!
