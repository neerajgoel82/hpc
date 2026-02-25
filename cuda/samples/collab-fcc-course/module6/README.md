# Module 6: CUDA APIs (cuBLAS, cuDNN, CUTLASS)

**Part of**: FreeCodeCamp CUDA Programming Course
**Duration**: 8-10 hours
**Difficulty**: Intermediate

---

## Overview

Module 6 teaches you how to leverage NVIDIA's highly optimized libraries for production performance. You'll master cuBLAS for linear algebra, cuDNN for deep learning operations, and explore CUTLASS for customizable GEMM kernels.

---

## Notebooks in This Module

### 01. cuBLAS: Half and Single Precision GEMM (`01_CUBLAS_01_cuBLAS_01_Hgemm_Sgemm.ipynb`)
**Duration**: 1.5 hours

**Learning Objectives**:
- Use cuBLAS for matrix multiplication
- Understand FP16 (half) vs FP32 (single) precision
- Compare performance across precisions
- Integrate cuBLAS into applications

**Key Concepts**:
- cuBLAS handle management
- `cublasHgemm` (FP16) and `cublasSgemm` (FP32)
- Column-major vs row-major layout
- Alpha/beta scaling parameters
- Leading dimensions (LD)

**Performance**:
- **FP16**: 15-25 TFLOPS on modern GPUs
- **FP32**: 10-15 TFLOPS on modern GPUs
- FP16 is 1.5-2x faster than FP32

**What You'll Build**: Performance comparison of different precisions

---

### 02. cuBLASLt: Advanced Matrix Operations (`01_CUBLAS_02_cuBLASLt_01_LtMatmul.ipynb`)
**Duration**: 2 hours

**Learning Objectives**:
- Use cuBLASLt for flexible GEMM
- Configure matrix layouts and operations
- Enable Tensor Core acceleration
- Optimize for specific hardware

**Key Concepts**:
- cuBLASLt API (more flexible than cuBLAS)
- Matrix descriptors and operation descriptors
- Algorithm selection and heuristics
- Tensor Core utilization
- Epilogue operations (bias, activation fusion)

**Advanced Features**:
- Automatic Tensor Core usage
- Fused operations (GEMM + bias + ReLU)
- Batched and strided operations

**What You'll Build**: Optimized GEMM with cuBLASLt

---

### 03. cuBLASLt: Performance Comparison (`01_CUBLAS_02_cuBLASLt_02_compare.ipynb`)
**Duration**: 1 hour

**Learning Objectives**:
- Benchmark cuBLAS vs cuBLASLt
- Understand when to use each API
- Measure Tensor Core speedups
- Optimize for your use case

**Key Comparisons**:
- cuBLAS (simple API) vs cuBLASLt (flexible API)
- Different algorithms and heuristics
- Tensor Core enabled vs disabled
- Various matrix sizes

**Performance Analysis**: Learn to choose the right tool

---

### 04. cuBLASXt: Multi-GPU Linear Algebra (`01_CUBLAS_03_cuBLASXt_01_demo.ipynb`)
**Duration**: 1.5 hours

**Learning Objectives**:
- Scale to multiple GPUs with cuBLASXt
- Understand automatic data partitioning
- Achieve near-linear scaling
- Handle large matrices beyond single GPU memory

**Key Concepts**:
- cuBLASXt API for multi-GPU
- Automatic workload distribution
- Pinned memory requirements
- Device selection and blocking

**Use Cases**:
- Matrices too large for single GPU
- Multi-GPU systems
- Maximum throughput requirements

**What You'll Build**: Multi-GPU GEMM application

---

### 05. cuBLASXt: Multi-GPU Comparison (`01_CUBLAS_03_cuBLASXt_02_compare.ipynb`)
**Duration**: 1 hour

**Learning Objectives**:
- Benchmark multi-GPU scaling
- Understand communication overhead
- Measure speedup vs single GPU
- Identify scaling bottlenecks

**Performance**: 1.8-2x speedup with 2 GPUs (ideally 2x)

---

### 06. cuDNN: Activation Functions (`02_CUDNN_00_Tanh.ipynb`)
**Duration**: 1 hour

**Learning Objectives**:
- Use cuDNN for neural network operations
- Implement activation functions (tanh)
- Understand tensor descriptors
- Compare with custom implementations

**Key Concepts**:
- cuDNN handle and tensor descriptors
- Activation forward and backward passes
- Memory layout (NCHW, NHWC)
- cuDNN performance advantages

**What You'll Build**: Tanh activation with cuDNN

---

### 07. cuDNN: 2D Convolution (`02_CUDNN_01_Conv2d_NCHW.ipynb`)
**Duration**: 2 hours

**Learning Objectives**:
- Implement 2D convolution with cuDNN
- Understand convolution parameters
- Work with NCHW tensor format
- Use cuDNN's optimized conv algorithms

**Key Concepts**:
- Convolution descriptors
- Filter descriptors
- cuDNN convolution algorithms (GEMM, Winograd, FFT)
- Workspace memory management
- Algorithm selection heuristics

**Applications**: Convolutional neural networks (CNNs)

**Performance**: 5-10x faster than naive implementation

---

### 08. cuDNN: Convolution Performance Comparison (`02_CUDNN_02_compare_conv.ipynb`)
**Duration**: 1 hour

**Learning Objectives**:
- Benchmark different cuDNN conv algorithms
- Understand algorithm trade-offs
- Measure workspace memory usage
- Select optimal algorithms for your case

**Algorithms Compared**:
- IMPLICIT_GEMM
- IMPLICIT_PRECOMP_GEMM
- GEMM
- DIRECT
- FFT
- WINOGRAD

**What You'll Learn**: When each algorithm is best

---

### 09. CUTLASS: Custom GEMM Templates (`optional_CUTLASS_compare.ipynb`)
**Duration**: 2 hours
**Optional**: Advanced topic

**Learning Objectives**:
- Use CUTLASS for customizable GEMM
- Understand template metaprogramming for CUDA
- Build high-performance custom kernels
- Compare with cuBLAS

**Key Concepts**:
- CUTLASS library architecture
- Template-based kernel generation
- Tile sizes and thread configurations
- Epilogue customization

**When to Use CUTLASS**:
- Need custom operations not in cuBLAS
- Fused kernels (GEMM + custom epilogue)
- Research and experimentation
- Maximum flexibility

---

## Learning Path

```
cuBLAS:
    01_Hgemm_Sgemm → 02_Lt_matmul → 02_Lt_compare
            ↓
    03_Xt_demo → 03_Xt_compare

cuDNN:
    02_Tanh → 02_Conv2d → 02_compare_conv

Optional:
    CUTLASS_compare
```

## Prerequisites

### Knowledge
- Completed Module 5
- Understanding of matrix operations
- Basic deep learning concepts (for cuDNN)
- Linear algebra fundamentals

### Technical
- Google Colab account
- T4 GPU or better (for best performance)

## Success Criteria

By completing Module 6, you should be able to:
- [ ] Use cuBLAS for matrix operations
- [ ] Choose between cuBLAS, cuBLASLt, cuBLASXt
- [ ] Implement neural network layers with cuDNN
- [ ] Optimize convolutions for CNNs
- [ ] Handle column-major layouts correctly
- [ ] Scale to multiple GPUs
- [ ] Integrate libraries into applications
- [ ] Achieve production-level performance

## Library Comparison

| Library | Use Case | Performance | Flexibility |
|---------|----------|-------------|-------------|
| **cuBLAS** | Standard BLAS ops | Very High | Low |
| **cuBLASLt** | Flexible GEMM | Very High | Medium |
| **cuBLASXt** | Multi-GPU BLAS | High | Low |
| **cuDNN** | Deep learning ops | Very High | Medium |
| **CUTLASS** | Custom GEMM | Very High | Very High |

## Performance Expectations

### GEMM Performance (4096x4096):
- **Custom CUDA (FP32)**: 10-15 TFLOPS
- **cuBLAS (FP32)**: 15-20 TFLOPS
- **cuBLAS (FP16)**: 20-30 TFLOPS
- **cuBLAS with Tensor Cores**: 100-300 TFLOPS

### Convolution (ResNet-50 layer):
- **Naive CUDA**: ~0.5 TFLOPS
- **cuDNN**: 5-10 TFLOPS (10-20x faster)

## Key Concepts

### cuBLAS Matrix Multiplication
```cpp
cublasHandle_t handle;
cublasCreate(&handle);

// C = alpha * A * B + beta * C
cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, k,
            &alpha, A, lda, B, ldb,
            &beta, C, ldc);

cublasDestroy(handle);
```

**Important**: cuBLAS uses column-major layout (Fortran convention)!

### cuBLASLt with Tensor Cores
```cpp
cublasLtHandle_t handle;
cublasLtCreate(&handle);

// Create descriptors
cublasLtMatmulDesc_t operationDesc;
cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;

// Configure and execute
cublasLtMatmul(handle, operationDesc,
               &alpha, A, Adesc, B, Bdesc,
               &beta, C, Cdesc, C, Cdesc,
               &algo, workspace, workspaceSize, stream);
```

### cuDNN Convolution
```cpp
cudnnHandle_t handle;
cudnnCreate(&handle);

// Create descriptors
cudnnTensorDescriptor_t inputDesc, outputDesc;
cudnnFilterDescriptor_t filterDesc;
cudnnConvolutionDescriptor_t convDesc;

// Find best algorithm
cudnnGetConvolutionForwardAlgorithm(...);

// Execute convolution
cudnnConvolutionForward(handle, &alpha,
                        inputDesc, input,
                        filterDesc, filter,
                        convDesc, algo, workspace, workspaceSize,
                        &beta, outputDesc, output);
```

## Common Pitfalls

1. **Column-Major Confusion**
   ```cpp
   // cuBLAS uses column-major!
   // For row-major C matrices, transpose:
   // C = A * B becomes C^T = B^T * A^T
   cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
               n, m, k, &alpha, B, ldb, A, lda, &beta, C, ldc);
   ```

2. **Workspace Memory**
   ```cpp
   // cuDNN algorithms need workspace
   // Query size first, then allocate
   cudnnGetConvolutionForwardWorkspaceSize(..., &workspaceSize);
   cudaMalloc(&workspace, workspaceSize);
   ```

3. **Descriptor Management**
   ```cpp
   // Always create and destroy descriptors
   cudnnCreateTensorDescriptor(&desc);
   // ... use descriptor ...
   cudnnDestroyTensorDescriptor(desc);
   ```

4. **Tensor Core Requirements**
   ```cpp
   // Tensor Cores need:
   // - FP16 or INT8 inputs
   // - Matrix dimensions multiple of 8 (FP16) or 16 (INT8)
   // - Proper alignment
   ```

## Optimization Tips

### 1. Choose the Right API
- **Simple BLAS operations**: Use cuBLAS
- **Need flexibility (epilogue fusion)**: Use cuBLASLt
- **Multiple GPUs**: Use cuBLASXt
- **Deep learning ops**: Use cuDNN
- **Custom requirements**: Use CUTLASS

### 2. Enable Tensor Cores
```cpp
// Use FP16 precision when possible
// Ensure matrix sizes are multiples of 8-16
// Use cuBLASLt or cuDNN for automatic Tensor Core usage
```

### 3. Workspace Optimization
```cpp
// Allocate workspace once, reuse
// Query sizes for all algorithms
// Choose algorithm balancing workspace vs performance
```

### 4. Algorithm Selection
```cpp
// Let cuDNN/cuBLAS choose automatically with heuristics
// Or benchmark and hardcode best algorithm
// Different algorithms excel at different sizes
```

## Time Estimate
- **Fast pace**: 1 week (8-10 hours)
- **Moderate pace**: 1.5 weeks (6-8 hours/week)
- **Relaxed pace**: 2 weeks (4-5 hours/week)

## Additional Resources

### Official Documentation
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [cuDNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/)
- [CUTLASS Repository](https://github.com/NVIDIA/cutlass)

### Tutorials
- [cuBLAS Tutorial](https://developer.nvidia.com/cublas)
- [cuDNN Developer Guide](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/)

### Course Resources
- [FreeCodeCamp CUDA Course](https://www.youtube.com/watch?v=86FAWCzIe_4)

## Practice Exercises

1. **GEMM variants**: Implement GEMV (matrix-vector), batched GEMM
2. **Layer implementation**: Build full CNN layer with cuDNN
3. **Multi-GPU scaling**: Benchmark scaling efficiency
4. **Algorithm comparison**: Test all cuDNN conv algorithms
5. **Custom fusion**: Use CUTLASS for custom epilogue

## Troubleshooting

### cuBLAS returns wrong results
**Solution**: Check if you're handling column-major layout correctly

### cuDNN out of memory
**Solution**: Reduce batch size or use CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM

### Tensor Cores not used
**Solution**: Verify FP16 precision, matrix sizes multiples of 8, cuBLASLt usage

### Slow performance
**Solution**: Profile, check algorithm selection, ensure Tensor Cores enabled

## Next Module

After completing Module 6, proceed to:
**Module 7: Optimizing Matrix Multiplication** - Deep dive into advanced optimization techniques and achieving peak performance.

**Path**: `../module7/README.md`

---

**Pro Tip**: Production applications almost always use cuBLAS and cuDNN rather than custom kernels. Learn these libraries well - they're heavily optimized and maintained by NVIDIA!

## Questions to Test Understanding

1. Why does cuBLAS use column-major layout?
2. What's the difference between cuBLAS and cuBLASLt?
3. When should you use cuBLASXt over cuBLAS?
4. How do you enable Tensor Cores in cuBLAS?
5. What are workspace requirements in cuDNN?
6. How do you select the best convolution algorithm?
7. When should you use CUTLASS instead of cuBLAS?
8. What precision gives the best performance on modern GPUs?

If you can use cuBLAS and cuDNN effectively and achieve near-peak performance, you're ready for Module 7! 🚀
