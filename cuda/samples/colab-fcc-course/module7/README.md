# Module 7: Optimizing Matrix Multiplication

**Part of**: FreeCodeCamp CUDA Programming Course
**Duration**: 2-3 hours
**Difficulty**: Advanced

---

## Overview

Module 7 dives deep into advanced optimization techniques for matrix multiplication. You'll learn loop unrolling, register optimization, and instruction-level parallelism to squeeze maximum performance from your kernels.

---

## Notebook in This Module

### Loop Unrolling for GEMM (`unrolling_example.ipynb`)
**Duration**: 2-3 hours

**Learning Objectives**:
- Understand loop unrolling benefits
- Implement unrolled GEMM kernels
- Apply instruction-level parallelism (ILP)
- Optimize register usage
- Achieve near-peak CUDA core performance

**Key Concepts**:
- **Loop unrolling**: Manually expand loop iterations
- **Register blocking**: Compute multiple outputs per thread
- **Instruction-level parallelism**: Keep pipeline full
- **Compiler optimization**: `#pragma unroll`
- **Memory access optimization**: Vectorized loads

**What You'll Build**: Highly optimized GEMM with unrolling

---

## What is Loop Unrolling?

### Without Unrolling:
```cuda
for (int k = 0; k < K; k++) {
    sum += A[row][k] * B[k][col];
}
```

### With Unrolling (4x):
```cuda
for (int k = 0; k < K; k += 4) {
    sum += A[row][k+0] * B[k+0][col];
    sum += A[row][k+1] * B[k+1][col];
    sum += A[row][k+2] * B[k+2][col];
    sum += A[row][k+3] * B[k+3][col];
}
```

**Benefits**:
- Reduces loop overhead
- Enables more instruction-level parallelism
- Better register utilization
- 1.5-2x speedup

---

## Optimization Techniques Covered

### 1. Basic Loop Unrolling
- Manually unroll inner loops
- Reduce branch instructions
- Keep ALU pipeline busy

**Performance**: 10-20% improvement

### 2. Register Blocking
Each thread computes multiple outputs (e.g., 4x4 tile):
```cuda
float results[4][4];
for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
        results[i][j] += ...;
    }
}
```

**Performance**: 2-4x improvement over naive

### 3. Vectorized Memory Access
Use `float4` for 4x coalescing:
```cuda
float4 vec = *reinterpret_cast<float4*>(&A[idx]);
// Process vec.x, vec.y, vec.z, vec.w
```

**Performance**: 1.5-2x bandwidth improvement

### 4. Instruction-Level Parallelism (ILP)
Interleave independent operations:
```cuda
// Instead of sequential:
a = load1(); b = load2(); c = a + b;

// Interleave loads and compute:
a = load1();
d = load3();  // Independent, can execute in parallel
b = load2();
e = load4();
c = a + b;
f = d + e;
```

**Performance**: Near-peak FLOPs utilization

### 5. Compiler Directives
```cuda
#pragma unroll
for (int i = 0; i < 8; i++) {
    sum += data[i];
}

// Or specify unroll factor
#pragma unroll 4
for (int i = 0; i < N; i++) {
    sum += data[i];
}
```

---

## GEMM Optimization Stages

```
Stage 0: Naive GEMM
  └─> 0.1-0.5 TFLOPS

Stage 1: + Shared Memory Tiling (Module 5)
  └─> 2-5 TFLOPS (10x)

Stage 2: + Register Blocking
  └─> 5-10 TFLOPS (2x)

Stage 3: + Loop Unrolling
  └─> 8-15 TFLOPS (1.5-2x)

Stage 4: + Vectorized Loads + ILP
  └─> 12-20 TFLOPS (1.5x)

cuBLAS (FP32):
  └─> 15-25 TFLOPS (reference)

cuBLAS with Tensor Cores (FP16):
  └─> 100-300 TFLOPS (10x more)
```

---

## Learning Path

```
Review Module 5 Tiled GEMM
        ↓
Learn Loop Unrolling Basics
        ↓
Implement Register Blocking
        ↓
Add Vectorized Loads
        ↓
Apply ILP Techniques
        ↓
Measure and Profile
        ↓
Compare with cuBLAS
```

## Prerequisites

### Knowledge
- Completed Modules 5-6
- Strong understanding of GEMM algorithm
- Comfortable with shared memory optimization
- Understanding of GPU architecture

### Technical
- Google Colab with T4 or better GPU
- Profiling tools familiarity

## Success Criteria

By completing Module 7, you should be able to:
- [ ] Implement loop unrolling manually
- [ ] Use `#pragma unroll` effectively
- [ ] Implement register blocking (4x4 or 8x8 tiles per thread)
- [ ] Use vectorized memory access (float4)
- [ ] Achieve 60-80% of cuBLAS FP32 performance
- [ ] Understand instruction-level parallelism
- [ ] Profile and identify bottlenecks
- [ ] Apply all optimization techniques learned

## Performance Expectations

### Target Performance (4096x4096 GEMM, FP32):

| Implementation | TFLOPS | % of cuBLAS |
|----------------|--------|-------------|
| **Naive** | 0.5 | 3% |
| **Tiled (Module 5)** | 5 | 25% |
| **+ Register blocking** | 10 | 50% |
| **+ Loop unrolling** | 15 | 75% |
| **+ Vectorization + ILP** | 18 | 90% |
| **cuBLAS (reference)** | 20 | 100% |

**Goal**: Achieve 60-80% of cuBLAS with custom kernel

---

## Key Code Patterns

### Register Blocking Pattern
```cuda
// Each thread computes 8x8 output tile
float results[8][8] = {0};

for (int k = 0; k < K; k += TILE_K) {
    // Load tiles to shared memory
    __syncthreads();
    
    // Compute 8x8 outputs
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            #pragma unroll
            for (int kk = 0; kk < TILE_K; kk++) {
                results[i][j] += tileA[i][kk] * tileB[kk][j];
            }
        }
    }
    __syncthreads();
}

// Store results
for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
        C[row+i][col+j] = results[i][j];
    }
}
```

### Vectorized Load Pattern
```cuda
// Load 4 floats at once (128-bit transaction)
float4* A_vec = reinterpret_cast<float4*>(A);
float4 values = A_vec[idx];

// Access individual elements
float a0 = values.x;
float a1 = values.y;
float a2 = values.z;
float a3 = values.w;
```

---

## Common Pitfalls

1. **Over-Unrolling**
   ```cuda
   // Too much unrolling increases register pressure
   // Reduces occupancy
   // Profile to find optimal unroll factor
   ```

2. **Misaligned Vectorized Loads**
   ```cuda
   // float4 requires 16-byte alignment
   // Ensure addresses are aligned
   assert((uintptr_t)ptr % 16 == 0);
   ```

3. **Register Spilling**
   ```cuda
   // Too many registers per thread
   // Causes spilling to local memory (slow!)
   // Check with: nvcc --ptxas-options=-v
   ```

4. **Ignoring Compiler Warnings**
   ```bash
   # Compile with warnings
   nvcc --ptxas-options=-v program.cu
   # Look for: "Used X registers, spill stores/loads"
   ```

## Profiling Commands

### Check Register Usage
```bash
nvcc --ptxas-options=-v unrolling_example.cu
# Look for: "Function properties for kernel"
# Registers: Want < 64 per thread for good occupancy
```

### Profile Performance
```bash
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed ./program
# Target: 60-80% for optimized GEMM
```

### Check for Spills
```bash
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,\
              l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum ./program
# Should be zero or very low
```

## Time Estimate
- **Fast pace**: 2-3 hours (focused session)
- **Moderate pace**: 4-6 hours (with experimentation)
- **Deep dive**: 8-10 hours (implement all variants)

## Additional Resources

### Papers
- "Anatomy of High-Performance Matrix Multiplication" (Goto & van de Geijn)
- "Optimizing Matrix Multiply on GPUs" (Nvidia)

### Documentation
- [CUDA C Programming Guide - Optimization](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#performance-guidelines)
- [CUDA C Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

### Advanced
- [CUTLASS](https://github.com/NVIDIA/cutlass) - NVIDIA's template library for GEMM
- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM)

## Practice Exercises

1. **Unroll factors**: Test 2x, 4x, 8x, 16x unrolling
2. **Register tiles**: Try 4x4, 8x8, 16x16 per thread
3. **Mixed unrolling**: Unroll outer loop, not inner
4. **Non-square matrices**: Optimize for M≠N≠K
5. **Benchmark suite**: Test various matrix sizes

## Debugging Tips

### Register Pressure Issues
```cuda
// Reduce register blocking size
// Instead of 8x8, try 4x4 or 6x6
float results[4][4];  // Uses fewer registers
```

### Performance Not Improving
```bash
# Profile to see actual bottleneck
ncu --set full -o report ./program
# Open report.ncu-rep in Nsight Compute GUI
```

### Numerical Errors
```cuda
// Unrolling shouldn't change results
// If results differ, check:
// 1. Boundary conditions
// 2. Initialization
// 3. Synchronization
```

## Next Module

After completing Module 7, proceed to:
**Module 8: Triton** - Learn high-level GPU programming with OpenAI's Triton language for easier kernel development.

**Path**: `../module8/README.md`

---

**Pro Tip**: Loop unrolling and register blocking are powerful but complex. Start simple, profile often, and incrementally add optimizations. You'll learn more from measuring than guessing!

## Questions to Test Understanding

1. What is loop unrolling and why does it help?
2. What is register blocking and how does it differ from shared memory tiling?
3. How do you load data using float4 and what are the requirements?
4. What is instruction-level parallelism (ILP)?
5. How do you check for register spilling?
6. What happens if you use too many registers per thread?
7. How close can you get to cuBLAS performance with custom kernels?
8. When should you use `#pragma unroll` vs manual unrolling?

If you can implement a highly optimized GEMM achieving 60-80% of cuBLAS performance, you've mastered CUDA optimization! 🚀
