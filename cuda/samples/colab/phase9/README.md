# Phase 9: Modern CUDA (Week 16+)

## Overview
Phase 9 covers cutting-edge CUDA features available on modern GPUs. You'll learn dynamic parallelism, CUDA graphs for low-overhead execution, mixed precision training, and tensor cores for extreme performance in matrix operations.

## Notebooks in This Phase

### 50_dynamic_parallelism.ipynb
**Duration**: 2.5 hours
**Learning Objectives**:
- Launch kernels from device code
- Implement recursive parallel algorithms
- Understand dynamic parallelism use cases
- Manage device-side synchronization

**Key Concepts**:
- Device-side kernel launches
- Nested parallelism
- Dynamic work creation
- Compilation with `nvcc -rdc=true`
- Compute capability 3.5+ required

**Applications**:
- Adaptive mesh refinement
- Recursive algorithms (quicksort)
- Tree traversal
- Dynamic workload generation

**Example**:
```cuda
__global__ void parent_kernel() {
    // Launch child kernel from device
    child_kernel<<<blocks, threads>>>();
    cudaDeviceSynchronize();
}
```

---

### 51_cuda_graphs.ipynb ⭐ KEY NOTEBOOK
**Duration**: 3 hours
**Learning Objectives**:
- Create and execute CUDA graphs
- Reduce kernel launch overhead by 10-100x
- Update graph parameters efficiently
- Understand graph capture and replay

**Key Concepts**:
- Graph creation (manual vs stream capture)
- Graph instantiation and execution
- Graph update (parameter changes)
- Launch overhead reduction: 5-10 μs → 50-100 ns
- Multi-kernel workflows

**Performance**: 10-100x lower launch overhead

**Use Cases**:
- Repeated execution of same workflow
- Real-time applications
- Inference pipelines
- Iterative algorithms

---

### 52_multi_process_service.ipynb
**Duration**: 2 hours
**Learning Objectives**:
- Use MPS for GPU sharing
- Run multiple processes on one GPU
- Understand context switching overhead
- Maximize GPU utilization

**Key Concepts**:
- Multi-Process Service (MPS)
- Context switching reduction
- GPU time-slicing
- Concurrent kernel execution
- Resource partitioning

**Benefits**:
- Multiple applications on one GPU
- Reduced context switching
- Better utilization for small kernels

---

### 53_mixed_precision_training.ipynb
**Duration**: 2.5 hours
**Learning Objectives**:
- Use FP16 for faster computation
- Implement mixed precision techniques
- Maintain numerical stability
- Apply loss scaling

**Key Concepts**:
- FP16 (half precision): 2x faster, 2x memory
- FP32 master weights
- Loss scaling for training stability
- Automatic mixed precision (AMP)
- Tensor Core utilization

**Performance**: 2-3x speedup on Volta+

---

### 54_tensor_cores_wmma_basics.ipynb ⭐ KEY NOTEBOOK
**Duration**: 3 hours
**Learning Objectives**:
- Use Tensor Cores via WMMA API
- Achieve massive speedups for matrix operations
- Understand matrix fragment operations
- Apply to neural network layers

**Key Concepts**:
- Warp Matrix Multiply-Accumulate (WMMA)
- Matrix fragments (16x16 tiles)
- Load/Store/MMA operations
- Volta/Ampere/Hopper Tensor Cores
- 100+ TFLOPS with Tensor Cores

**Requirements**:
- Compute capability 7.0+ (Volta, Turing, Ampere, Hopper)
- Matrix sizes must be multiples of 16

**Example**:
```cuda
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

wmma::load_matrix_sync(a_frag, a, 16);
wmma::load_matrix_sync(b_frag, b, 16);
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);
```

---

### 55_wmma_gemm_full_implementation.ipynb ⭐ KEY NOTEBOOK
**Duration**: 4 hours
**Learning Objectives**:
- Implement complete GEMM with Tensor Cores
- Achieve 100+ TFLOPS on modern GPUs
- Combine WMMA with shared memory tiling
- Optimize for maximum performance

**Key Concepts**:
- Multi-level tiling with WMMA
- Shared memory staging
- WMMA fragment management
- Achieving peak Tensor Core throughput
- Mixed precision accumulation

**Performance Targets**:
- **V100**: 100-120 TFLOPS (FP16)
- **A100**: 250-300 TFLOPS (FP16)
- **H100**: 800-1000 TFLOPS (FP8)

**Result**: 10-20x faster than FP32 GEMM

---

## Learning Path

```
50-dynamic-parallelism
        ↓
51-cuda-graphs ⭐
        ↓
52-mps-demo
        ↓
53-mixed-precision
        ↓
54-tensor-cores-basics ⭐
        ↓
55-wmma-gemm ⭐
```

## Prerequisites
- Completed Phases 1-8
- Strong understanding of all CUDA concepts
- Modern GPU (Volta/Turing/Ampere/Hopper) for features 53-55
- Advanced optimization skills

## Success Criteria

By the end of Phase 9, you should be able to:
- [ ] Use dynamic parallelism for recursive algorithms
- [ ] Create and execute CUDA graphs
- [ ] Apply mixed precision techniques
- [ ] Program Tensor Cores via WMMA
- [ ] Implement GEMM achieving 100+ TFLOPS
- [ ] Understand modern GPU architectures
- [ ] Choose appropriate modern features for problems
- [ ] Achieve state-of-the-art performance

## Modern GPU Features

### Feature Availability by Compute Capability:

| Feature | Compute Cap | GPUs |
|---------|-------------|------|
| **Dynamic Parallelism** | 3.5+ | Kepler and later |
| **CUDA Graphs** | 7.0+ | All modern GPUs |
| **MPS** | 3.5+ | Kepler and later |
| **Tensor Cores (FP16)** | 7.0+ | Volta, Turing, Ampere, Hopper |
| **Tensor Cores (INT8)** | 7.2+ | Turing, Ampere, Hopper |
| **Tensor Cores (FP8)** | 8.9+ | Hopper (H100) |

### Check Your GPU:
```cuda
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
```

## Performance Comparison

### GEMM Performance (M=N=K=4096):

| Implementation | TFLOPS | Notes |
|----------------|--------|-------|
| **Naive CUDA** | 0.1-0.5 | No optimization |
| **Optimized CUDA (FP32)** | 10-15 | Phase 5 techniques |
| **cuBLAS (FP32)** | 15-20 | NVIDIA optimized |
| **Optimized CUDA (FP16)** | 20-30 | 2x speedup |
| **Tensor Cores (FP16)** | 100-300 | 10x+ speedup |
| **cuBLAS with Tensor Cores** | 100-400 | Near-peak |

**Speedup**: 100-200x vs naive implementation!

## CUDA Graphs Deep Dive

### Why CUDA Graphs?

**Problem**: Kernel launch overhead is 5-10 microseconds
- For small kernels, launch overhead dominates
- Iterative algorithms waste time on launches

**Solution**: CUDA Graphs
- Create graph once, execute many times
- Launch overhead: ~50-100 nanoseconds
- **100x lower overhead!**

### Creating Graphs:

#### Method 1: Manual Construction
```cuda
cudaGraph_t graph;
cudaGraphCreate(&graph, 0);

// Add kernel node
cudaKernelNodeParams params;
params.func = (void*)kernel;
params.gridDim = grid;
params.blockDim = block;
cudaGraphNode_t node;
cudaGraphAddKernelNode(&node, graph, NULL, 0, &params);

// Instantiate and launch
cudaGraphExec_t instance;
cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
cudaGraphLaunch(instance, stream);
```

#### Method 2: Stream Capture (Easier)
```cuda
cudaGraph_t graph;
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

// Record operations
kernel1<<<grid, block, 0, stream>>>();
kernel2<<<grid, block, 0, stream>>>();

cudaStreamEndCapture(stream, &graph);

// Instantiate and launch
cudaGraphExec_t instance;
cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
cudaGraphLaunch(instance, stream);  // Fast!
```

### When to Use Graphs:
✅ Repeated execution of same workflow
✅ Many small kernels
✅ Low-latency applications
✅ Inference pipelines
❌ Single-shot computation
❌ Highly dynamic workflows

## Tensor Cores Programming

### Matrix Sizes for WMMA:
- Must be multiples of 16 (16x16 tiles)
- Supported shapes: 16x16x16, 32x8x16, 8x32x16
- Best performance: large matrices (4096x4096+)

### WMMA Fragment Types:
```cuda
// Matrix A fragment (input)
wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;

// Matrix B fragment (input)
wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b_frag;

// Accumulator fragment (output, higher precision)
wmma::fragment<wmma::accumulator, M, N, K, float> acc_frag;
```

### WMMA Operations:
```cuda
// Load from memory to fragment
wmma::load_matrix_sync(a_frag, a_ptr, lda);

// Matrix multiply-accumulate: acc = A * B + acc
wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

// Store fragment to memory
wmma::store_matrix_sync(c_ptr, acc_frag, ldc, wmma::mem_row_major);
```

### Tensor Core GEMM Pattern:
```
For each 16x16 output tile:
  1. Load A tile (16x16) into fragment
  2. Load B tile (16x16) into fragment
  3. Multiply-accumulate with wmma::mma_sync
  4. Store result tile (16x16)
  
Combine with shared memory tiling for large matrices
```

## Mixed Precision Training

### Why Mixed Precision?
- **FP16**: 2x faster compute, 2x memory reduction
- **FP32**: Better numeric stability
- **Strategy**: Compute in FP16, keep master weights in FP32

### Loss Scaling:
```cuda
// Scale loss to prevent underflow in FP16
loss = loss * scale_factor;  // e.g., scale_factor = 1024

// Backward pass in FP16
compute_gradients(loss);

// Unscale gradients
gradients = gradients / scale_factor;

// Update FP32 master weights
update_weights_fp32(gradients);

// Convert to FP16 for next forward pass
weights_fp16 = convert_to_fp16(weights_fp32);
```

## Common Pitfalls

1. **Dynamic Parallelism Overhead**
   ```cuda
   // Device-side launches are expensive
   // Only use for truly dynamic workloads
   // Consider alternatives first
   ```

2. **Graph Limitations**
   ```cuda
   // Cannot change topology during execution
   // Can only update parameters
   // Not suitable for very dynamic algorithms
   ```

3. **Tensor Core Alignment**
   ```cuda
   // Matrix sizes must be multiples of 16
   // Pointers must be 16-byte aligned
   // Leading dimensions must be multiples of 16
   ```

4. **Mixed Precision Stability**
   ```cuda
   // Some operations need FP32 (e.g., batch norm)
   // Use loss scaling to prevent underflow
   // Validate numerical correctness
   ```

## Time Estimate
- **Fast pace**: 2 weeks (4-5 hours/day)
- **Moderate pace**: 2.5 weeks (3 hours/day)
- **Relaxed pace**: 3-4 weeks (2 hours/day)

## Additional Resources

### NVIDIA Documentation
- [Dynamic Parallelism Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-dynamic-parallelism)
- [CUDA Graphs Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)
- [Tensor Cores Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)
- [Mixed Precision Training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/)

### Papers & Talks
- "Programming Tensor Cores in CUDA 9" (NVIDIA Blog)
- "Mixed Precision Training" (Micikevicius et al.)
- "CUDA Graphs: Performance Optimization" (GTC talks)

### Tools
- [CUTLASS](https://github.com/NVIDIA/cutlass) - CUDA Templates for Linear Algebra
- [Automatic Mixed Precision](https://developer.nvidia.com/automatic-mixed-precision) - PyTorch/TensorFlow

## Practice Exercises

1. **Recursive quicksort** with dynamic parallelism
2. **LSTM inference** using CUDA graphs
3. **ResNet forward pass** with mixed precision
4. **Custom GEMM** with Tensor Cores achieving 100+ TFLOPS
5. **Attention mechanism** (Transformers) with WMMA

## Future Topics

### Beyond Phase 9:
- **CUDA on ARM**: Grace CPU + Hopper GPU
- **Multi-instance GPU (MIG)**: Partition A100/H100
- **CUDA in Python**: CuPy, Numba, PyTorch custom ops
- **Deep Learning**: cuDNN, TensorRT
- **HPC Libraries**: cuSOLVER, cuTENSOR
- **Quantum Computing**: CUDA Quantum

## Next Steps

**Congratulations!** 🎉 You've completed all 9 phases!

### Continue Your CUDA Journey:

1. **Build Projects**: Apply skills to your domain
2. **Contribute**: Open source CUDA projects
3. **Specialize**: Deep learning, HPC, graphics, etc.
4. **Stay Updated**: Follow NVIDIA GTC, blogs, papers
5. **Teach Others**: Solidify your knowledge

### Career Paths:
- GPU Computing Engineer
- Machine Learning Engineer
- HPC Developer
- Graphics Programmer
- Research Scientist

### Keep Learning:
- Advanced courses (CUDA Training Series)
- Research papers (NVIDIA Research)
- Open source (RAPIDS, CUTLASS, Thrust)
- Community (NVIDIA Forums, Stack Overflow)

---

**Pro Tip**: Modern GPU features like Tensor Cores can provide 10-100x speedups over traditional CUDA. Master them for cutting-edge performance!

## Questions to Test Your Understanding

1. When should you use dynamic parallelism vs static?
2. What's the overhead reduction from CUDA graphs?
3. How does mixed precision maintain numerical stability?
4. What are the matrix size requirements for WMMA?
5. How do Tensor Cores achieve 100+ TFLOPS?
6. What's the difference between graph creation methods?
7. When should you use FP16 vs FP32?
8. How do you update parameters in a CUDA graph?

If you can answer these and implement Tensor Core GEMM, you've mastered CUDA! 🚀

---

## Final Thoughts

You've learned:
- ✅ CUDA fundamentals (Phases 1-2)
- ✅ Optimization techniques (Phases 3-4)
- ✅ Advanced algorithms (Phase 5)
- ✅ Concurrency and multi-GPU (Phase 6)
- ✅ Performance engineering (Phase 7)
- ✅ Real applications (Phase 8)
- ✅ Modern GPU features (Phase 9)

**You're now a CUDA expert!** Go build amazing GPU-accelerated applications! 💪
