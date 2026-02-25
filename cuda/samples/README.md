# CUDA Samples - Comprehensive Learning Repository

**67 complete CUDA programs** covering everything from basics to tensor cores.

---

## Quick Start

### Local Development (.cu files)
```bash
cd local/phase1
nvcc -arch=sm_70 03_vector_add.cu -o vector_add
./vector_add
```

### Google Colab (Notebooks)
1. Go to https://colab.research.google.com
2. Upload a notebook from `colab/notebooks/`
3. Runtime → Change runtime type → GPU (T4, V100, or A100)
4. Run all cells

---

## Repository Structure

```
cuda/samples/
├── local/              # 67 .cu files for native GPU execution
│   ├── phase1/        # 7 files  - Foundations
│   ├── phase2/        # 7 files  - Memory Management
│   ├── phase3/        # 9 files  - Optimization (warp ops, tiling)
│   ├── phase4/        # 7 files  - Advanced Memory
│   ├── phase5/        # 7 files  - Advanced Algorithms (GEMM, sort)
│   ├── phase6/        # 7 files  - Streams & Concurrency
│   ├── phase7/        # 6 files  - Performance Engineering
│   ├── phase8/        # 10 files - Real Applications (N-body, ML, finance)
│   └── phase9/        # 7 files  - Modern CUDA (graphs, tensor cores)
│
└── colab/
    └── notebooks/      # 56 notebooks for Google Colab
        ├── phase1/     # Foundations
        ├── phase2/     # Memory
        ├── phase3/     # Optimization
        ├── phase4/     # Advanced Memory
        ├── phase5/     # Advanced Algorithms
        ├── phase6/     # Concurrency
        ├── phase7/     # Performance
        ├── phase8/     # Applications
        └── phase9/     # Modern CUDA
```

---

## What's Included

### Phase 1: Foundations (7 programs)
- Device queries, vector add, matrix add
- Thread indexing patterns (1D, 2D, grid-stride)

### Phase 2: Memory Management (7 programs)
- Host-device transfers, bandwidth benchmarking
- Pinned vs pageable memory
- Unified memory, shared memory basics

### Phase 3: Optimization (9 programs)
- **Tiled matrix multiplication** (16x16 shared memory)
- **Warp shuffle** (`__shfl_down_sync`)
- Occupancy tuning, parallel reduction
- Prefix sum (scan), histogram

### Phase 4: Advanced Memory (7 programs)
- Texture memory, constant memory
- Zero-copy (mapped) memory
- Atomic operations, cooperative groups

### Phase 5: Advanced Algorithms (7 programs)
- **Optimized GEMM** (tiled with shared memory)
- cuBLAS integration
- Matrix transpose (bank conflict avoidance)
- Bitonic sort, radix sort, Thrust library

### Phase 6: Streams & Concurrency (7 programs)
- CUDA streams for parallel execution
- Asynchronous pipelines
- Event-based timing, multi-GPU basics

### Phase 7: Performance Engineering (6 programs)
- Profiling with nvprof/Nsight
- Debugging patterns, kernel fusion
- Fast math intrinsics

### Phase 8: Real Applications (10 programs)
- **cuFFT**: Fast Fourier Transform
- **cuSPARSE**: Sparse matrix operations
- **cuRAND**: Random number generation
- **Image Processing**: Gaussian blur, Sobel edge detection
- **Ray Tracer**: Sphere intersection with shading
- **N-body Simulation**: Gravitational forces (F=Gm₁m₂/r²)
- **Neural Network**: Forward/backward propagation
- **Molecular Dynamics**: Lennard-Jones potential
- **Option Pricing**: Monte Carlo with Black-Scholes

### Phase 9: Modern CUDA (7 programs)
- **Dynamic Parallelism**: Child kernel launches
- **CUDA Graphs**: Low-overhead execution
- **MPS**: Multi-Process Service
- **Mixed Precision**: FP16/FP32 operations
- **Tensor Cores**: WMMA API
- **WMMA GEMM**: Complete tensor core matrix multiply

---

## Compilation Guide

### Basic Programs
```bash
nvcc -arch=sm_70 program.cu -o program
./program
```

### With Libraries
```bash
# cuBLAS
nvcc -arch=sm_70 25_cublas_integration.cu -o cublas -lcublas

# cuFFT
nvcc -arch=sm_70 41_cufft_demo.cu -o fft -lcufft

# cuSPARSE
nvcc -arch=sm_70 42_cusparse_demo.cu -o sparse -lcusparse

# cuRAND
nvcc -arch=sm_70 43_curand_demo.cu -o rand -lcurand
```

### Dynamic Parallelism
```bash
nvcc -arch=sm_35 -rdc=true 50_dynamic_parallelism.cu -o cdp -lcudadevrt
```

### Compute Capability
Choose based on your GPU:
- **sm_60**: Pascal (GTX 1080, Tesla P100)
- **sm_70**: Volta (Tesla V100, T4)
- **sm_75**: Turing (RTX 2080)
- **sm_80**: Ampere (A100, RTX 3090)
- **sm_86**: Ampere (RTX 3060)
- **sm_89**: Ada Lovelace (RTX 4090)
- **sm_90**: Hopper (H100)

---

## Learning Paths

### Beginner (2-3 weeks): Phases 1-3
**Week 1**: Basics - Device queries, vector/matrix operations, thread indexing
**Week 2**: Memory - Transfers and bandwidth, shared memory
**Week 3**: Optimization - Matrix tiling, warp operations, parallel algorithms

### Intermediate (3-4 weeks): Phases 4-6
**Week 4**: Advanced Memory - Texture, constant, atomics
**Week 5**: Algorithms - Optimized GEMM, libraries, sorting
**Week 6-7**: Concurrency - Streams, async operations, multi-GPU

### Advanced (3-4 weeks): Phases 7-9
**Week 8**: Performance - Profiling, debugging, kernel fusion
**Week 9-10**: Applications - Physics, ML, finance
**Week 11**: Modern CUDA - Dynamic parallelism, graphs, tensor cores

**Total**: 8-11 weeks for complete mastery

---

## Notebook Structure

Each notebook has:

**Example Cell (Cell 3)**: ✅ Complete working implementation
**Exercise Cell (Cell 5)**: 📝 Template for practice

Learn by example, then learn by doing!

---

## Hardware Requirements

### Minimum
- CUDA-capable GPU (compute capability 3.5+)
- CUDA Toolkit 11.0+
- Latest drivers

### Feature Requirements
| Feature | Min. Compute | Example GPUs |
|---------|--------------|--------------|
| Basics | sm_30 | Any modern GPU |
| Dynamic Parallelism | sm_35 | Kepler (K80) or newer |
| Tensor Cores | sm_70 | Volta (V100), Turing (T4, RTX 20xx), Ampere (A100, RTX 30xx) |

---

## Testing Your Setup

```bash
# Test compilation
cd local/phase1
nvcc -arch=sm_70 01_hello_world.cu -o hello && ./hello

# Test with timing
nvcc -arch=sm_70 03_vector_add.cu -o vec_add && ./vec_add

# Test libraries
cd ../phase8
nvcc -arch=sm_70 41_cufft_demo.cu -o fft -lcufft && ./fft

# Test tensor cores (requires sm_70+)
cd ../phase9
nvcc -arch=sm_70 54_tensor_cores.cu -o tensor && ./tensor
```

---

## Key Concepts Demonstrated

- **Memory**: Coalescing, shared memory, texture/constant memory
- **Optimization**: Tiling, warp operations, occupancy tuning
- **Algorithms**: GEMM, reduction, scan, sorting
- **Libraries**: cuBLAS, cuFFT, cuSPARSE, cuRAND, Thrust
- **Modern**: Dynamic parallelism, graphs, tensor cores

---

## Additional Resources

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [cuFFT Documentation](https://docs.nvidia.com/cuda/cufft/)

---

## Statistics

- **Total Programs**: 67 .cu files + 56 notebooks
- **Lines of Code**: ~21,000
- **Concepts Covered**: 80+
- **Learning Time**: 8-11 weeks for complete mastery

---

**Ready to start?** Jump to `local/phase1/` and compile your first CUDA program! 🚀
