# CUDA Samples - Comprehensive Learning Repository

**67 complete CUDA programs** covering everything from basics to tensor cores.

**56 interactive notebooks** for Google Colab with working examples and exercises.

---

## Quick Start

### Option 1: Local Development (.cu files)
```bash
cd local/phase1
nvcc -arch=sm_70 03_vector_add.cu -o vector_add
./vector_add
```

### Option 2: Google Colab (No GPU Required)
1. Go to https://colab.research.google.com
2. Upload a notebook from `colab/notebooks/phase1/`
3. Runtime → Change runtime type → GPU (T4, V100, or A100)
4. Run all cells and start learning!

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
        ├── phase1/     # 6 notebooks (00-05) - Foundations
        ├── phase2/     # 6 notebooks (06-11) - Memory
        ├── phase3/     # 6 notebooks (12-17) - Optimization
        ├── phase4/     # 6 notebooks (18-23) - Advanced Memory
        ├── phase5/     # 6 notebooks (24-29) - Advanced Algorithms
        ├── phase6/     # 6 notebooks (30-35) - Concurrency
        ├── phase7/     # 5 notebooks (36-40) - Performance
        ├── phase8/     # 9 notebooks (41-49) - Applications
        └── phase9/     # 6 notebooks (50-55) - Modern CUDA
```

---

## What's Included

### Phase 1: Foundations (7 programs / 6 notebooks)
**Learning Objectives**: Basic CUDA programming, thread hierarchy, memory transfers

**Programs**:
- 00: Setup Verification ⭐ START HERE
- 01: Hello World
- 02: Device Query
- 03: Vector Addition
- 04: Matrix Addition
- 05: Thread Indexing (1D, 2D, grid-stride)

**Skills**: Launch kernels, understand threads/blocks/grids, allocate GPU memory

---

### Phase 2: Memory Management (7 programs / 6 notebooks)
**Learning Objectives**: Memory types, bandwidth optimization, shared memory

**Programs**:
- 06: Memory Basics & Transfers
- 07: Bandwidth Benchmarking
- 08: Unified Memory (Managed Memory)
- 09: Shared Memory Basics
- 10: Tiled Matrix Multiplication
- 11: Memory Coalescing Patterns

**Skills**: Optimize memory access, use shared memory, understand coalescing

---

### Phase 3: Optimization Fundamentals (9 programs / 6 notebooks)
**Learning Objectives**: Warp-level operations, parallel algorithms, occupancy

**Programs**:
- 12: Warp Divergence Analysis
- 13: **Warp Shuffle** (`__shfl_down_sync`)
- 14: Occupancy Tuning
- 15: **Parallel Reduction** (sum, max, min)
- 16: Prefix Sum (Scan)
- 17: Histogram with Atomics

**Skills**: Use warp operations, implement reduction, optimize occupancy

---

### Phase 4: Advanced Memory & Synchronization (7 programs / 6 notebooks)
**Learning Objectives**: Specialized memory, atomics, cooperative groups

**Programs**:
- 18: Texture Memory
- 19: Constant Memory
- 20: Zero-Copy (Mapped Memory)
- 21: Atomic Operations
- 22: Cooperative Groups
- 23: Multi-Kernel Synchronization

**Skills**: Use texture/constant memory, atomic ops, cooperative groups

---

### Phase 5: Advanced Algorithms (7 programs / 6 notebooks)
**Learning Objectives**: Optimized GEMM, CUDA libraries, sorting

**Programs**:
- 24: **Optimized GEMM** (tiled with 16x16 shared memory)
- 25: cuBLAS Integration
- 26: Matrix Transpose (bank conflict avoidance)
- 27: Bitonic Sort
- 28: Radix Sort
- 29: Thrust Library Examples

**Skills**: Implement GEMM, use cuBLAS/Thrust, optimize algorithms

---

### Phase 6: Streams & Concurrency (7 programs / 6 notebooks)
**Learning Objectives**: Asynchronous execution, multi-GPU programming

**Programs**:
- 30: CUDA Streams Basics
- 31: Async Pipeline (overlap compute/transfer)
- 32: Events and Timing
- 33: Multi-GPU Basics
- 34: Peer-to-Peer Transfer
- 35: NCCL Collectives

**Skills**: Use streams, overlap operations, program multiple GPUs

---

### Phase 7: Performance Engineering (6 programs / 5 notebooks)
**Learning Objectives**: Profiling, debugging, advanced optimization

**Programs**:
- 36: Profiling with Nsight
- 37: Debugging CUDA Programs
- 38: **Kernel Fusion** (performance comparison)
- 39: Fast Math Intrinsics
- 40: Advanced Optimization Techniques

**Skills**: Profile with Nsight, debug effectively, apply optimizations

---

### Phase 8: Real-World Applications (10 programs / 9 notebooks)
**Learning Objectives**: Domain libraries, complete applications

**Programs**:
- 41: **cuFFT** - Fast Fourier Transform
- 42: **cuSPARSE** - Sparse matrix operations
- 43: **cuRAND** - Random number generation
- 44: **Image Processing** - Gaussian blur, Sobel edge detection
- 45: **Ray Tracer** - Sphere intersection with shading
- 46: **N-body Simulation** - Gravitational forces (F=Gm₁m₂/r²)
- 47: **Neural Network** - Forward/backward propagation
- 48: **Molecular Dynamics** - Lennard-Jones potential
- 49: **Option Pricing** - Monte Carlo with Black-Scholes

**Skills**: Use domain libraries, build complete applications

---

### Phase 9: Modern CUDA (7 programs / 6 notebooks)
**Learning Objectives**: Latest CUDA features, tensor cores

**Programs**:
- 50: **Dynamic Parallelism** - Child kernel launches
- 51: **CUDA Graphs** - Low-overhead execution
- 52: **MPS** - Multi-Process Service
- 53: **Mixed Precision** - FP16/FP32 operations
- 54: **Tensor Cores** - WMMA API basics
- 55: **WMMA GEMM** - Complete tensor core matrix multiply (400+ lines)

**Skills**: Use modern features, program tensor cores, optimize for latest GPUs

---

## Learning Paths

### Path 1: Complete Beginner (8-11 weeks)
**Follow sequentially from Phase 1 → Phase 9**

| Weeks | Phases | Focus |
|-------|--------|-------|
| 1-2 | Phase 1 | Basics - Device queries, kernels, memory transfers |
| 3-4 | Phase 2 | Memory - Shared memory, coalescing, bandwidth |
| 5-6 | Phase 3 | Optimization - Tiling, warp ops, reduction |
| 7-8 | Phases 4-5 | Advanced - Texture/constant memory, GEMM, libraries |
| 9 | Phase 6 | Concurrency - Streams, async, multi-GPU |
| 10 | Phase 7 | Performance - Profiling, debugging, optimization |
| 11 | Phases 8-9 | Applications - Real projects, modern features |

**Time commitment**: 1-2 hours/day

---

### Path 2: Experienced Programmer (6-8 weeks)
Skip basics, focus on GPU-specific concepts:
- Quick review: Phase 1 (2-3 days)
- Core focus: Phases 2-5 (4-5 weeks)
- Advanced: Phases 6-9 (2-3 weeks)

**Time commitment**: 2-3 hours/day

---

### Path 3: Specific Topics (Custom)
Jump to relevant phases based on your needs:

- **Memory optimization** → Phase 2, 4
- **Performance tuning** → Phase 3, 7
- **Algorithm implementation** → Phase 5
- **Real applications** → Phase 8
- **Modern GPU features** → Phase 9
- **Multi-GPU programming** → Phase 6

---

## Compilation Guide

### Basic Programs
```bash
nvcc -arch=sm_70 program.cu -o program
./program
```

### Programs Using CUDA Libraries
```bash
# cuBLAS (matrix operations)
nvcc -arch=sm_70 25_cublas_integration.cu -o cublas -lcublas

# cuFFT (Fast Fourier Transform)
nvcc -arch=sm_70 41_cufft_demo.cu -o fft -lcufft

# cuSPARSE (sparse matrices)
nvcc -arch=sm_70 42_cusparse_demo.cu -o sparse -lcusparse

# cuRAND (random numbers)
nvcc -arch=sm_70 43_curand_demo.cu -o rand -lcurand
```

### Dynamic Parallelism (Phase 9)
```bash
nvcc -arch=sm_35 -rdc=true 50_dynamic_parallelism.cu -o cdp -lcudadevrt
```

### Compute Capability Reference
Choose `-arch=sm_XX` based on your GPU:

| Architecture | Compute Capability | Example GPUs |
|--------------|-------------------|--------------|
| Pascal | sm_60, sm_61 | GTX 1080, Tesla P100 |
| Volta | sm_70 | Tesla V100, Titan V |
| Turing | sm_75 | RTX 2080, T4, Quadro RTX |
| Ampere | sm_80, sm_86 | A100, RTX 3090, RTX 3060 |
| Ada Lovelace | sm_89 | RTX 4090, RTX 4080 |
| Hopper | sm_90 | H100 |

**Tip**: Use `nvcc -arch=sm_70` as default (works on Volta, Turing, Ampere)

---

## Hardware Requirements

### Minimum Requirements
- CUDA-capable GPU (compute capability 3.5+)
- CUDA Toolkit 11.0+ (12.0+ recommended)
- Latest NVIDIA drivers
- C++11 compatible compiler

### Feature-Specific Requirements
| Feature | Min. Compute | Phases |
|---------|--------------|--------|
| Basic CUDA | sm_30 | 1-7 |
| Dynamic Parallelism | sm_35 | 9 |
| Cooperative Groups | sm_60 | 4 |
| Tensor Cores | sm_70 | 9 |
| WMMA API | sm_70 | 9 |

### Google Colab GPUs
- **T4** (sm_75) - Free tier, perfect for learning
- **V100** (sm_70) - Sometimes available
- **A100** (sm_80) - Colab Pro

---

## Testing Your Setup

### Test 1: Basic Compilation
```bash
cd local/phase1
nvcc -arch=sm_70 01_hello_world.cu -o hello && ./hello
```
**Expected output**: "Hello from GPU! Block X, Thread Y"

### Test 2: Vector Operations
```bash
nvcc -arch=sm_70 03_vector_add.cu -o vec_add && ./vec_add
```
**Expected output**: Performance metrics and verification

### Test 3: CUDA Libraries
```bash
cd ../phase8
nvcc -arch=sm_70 41_cufft_demo.cu -o fft -lcufft && ./fft
```
**Expected output**: FFT computation results

### Test 4: Tensor Cores (requires sm_70+)
```bash
cd ../phase9
nvcc -arch=sm_70 54_tensor_cores.cu -o tensor && ./tensor
```
**Expected output**: WMMA operation results

---

## Notebook Structure

Each Colab notebook contains:

### Cell 3: Example Implementation ✅
- **Complete working CUDA code**
- Real implementations (not templates)
- Run to see the concept in action
- Examples: N-body with gravity, WMMA with tensor cores

### Cell 5: Exercise Template 📝
- **Practice template for hands-on learning**
- Implement the concept yourself
- Compare with Cell 3 example
- Learn by doing!

**Learning workflow**:
1. Read concept explanation (Cell 1-2)
2. Study and run example (Cell 3)
3. Try implementing yourself (Cell 5)
4. Compare your solution with example
5. Take notes (Cell 8)

---

## Success Criteria

### ✓ Phase 1 Complete
- [ ] Can write and launch CUDA kernels
- [ ] Understand thread hierarchy (blocks, threads, grids)
- [ ] Allocate and transfer GPU memory
- [ ] Implement basic parallel operations

### ✓ Phase 2 Complete
- [ ] Master different memory types
- [ ] Use shared memory effectively
- [ ] Optimize memory access patterns
- [ ] Understand and achieve coalescing

### ✓ Phase 3 Complete
- [ ] Understand warp-level execution
- [ ] Implement reduction algorithms
- [ ] Use warp shuffle operations
- [ ] Optimize kernel occupancy

### ✓ Phase 4 Complete
- [ ] Use texture and constant memory
- [ ] Master atomic operations
- [ ] Use cooperative groups
- [ ] Synchronize multiple kernels

### ✓ Phase 5 Complete
- [ ] Implement optimized matrix multiply (GEMM)
- [ ] Use CUDA libraries (cuBLAS, Thrust)
- [ ] Implement sorting algorithms
- [ ] Optimize complex algorithms

### ✓ Phase 6 Complete
- [ ] Use CUDA streams for concurrency
- [ ] Overlap compute and memory transfers
- [ ] Program multiple GPUs
- [ ] Use NCCL for multi-GPU communication

### ✓ Phase 7 Complete
- [ ] Profile with Nsight tools
- [ ] Debug CUDA applications effectively
- [ ] Apply advanced optimizations
- [ ] Measure and improve performance

### ✓ Phase 8 Complete
- [ ] Use domain-specific libraries
- [ ] Build complete applications
- [ ] Integrate multiple techniques
- [ ] Solve real-world problems

### ✓ Phase 9 Complete
- [ ] Use modern CUDA features
- [ ] Program tensor cores with WMMA
- [ ] Use CUDA graphs
- [ ] Master mixed precision computing

---

## Key Concepts Demonstrated

### Memory Hierarchy & Optimization
- Memory coalescing (aligned, contiguous access)
- Shared memory (on-chip, low latency)
- Texture memory (cached, read-only)
- Constant memory (broadcast to all threads)
- Unified/managed memory (automatic migration)

### Parallel Algorithms
- Parallel reduction (sum, max, min)
- Prefix sum (scan)
- Matrix multiplication (GEMM) with tiling
- Sorting (bitonic, radix)
- Histogram with atomics

### Performance Optimization
- Tiling (shared memory blocking)
- Warp operations (shuffle, vote)
- Occupancy tuning (threads per SM)
- Kernel fusion (reduce overhead)
- Bank conflict avoidance

### CUDA Libraries
- cuBLAS (dense linear algebra)
- cuFFT (Fast Fourier Transform)
- cuSPARSE (sparse matrices)
- cuRAND (random numbers)
- Thrust (STL for CUDA)
- NCCL (multi-GPU communication)

### Modern Features
- Dynamic parallelism (GPU-side kernels)
- CUDA graphs (low-overhead execution)
- Tensor cores (WMMA API)
- Mixed precision (FP16/FP32)
- Cooperative groups

---

## Additional Resources

### Official Documentation
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [cuFFT Documentation](https://docs.nvidia.com/cuda/cufft/)

### Tools
- **Nsight Compute** - Kernel profiling
- **Nsight Systems** - System-wide profiling
- **cuda-gdb** - CUDA debugger
- **compute-sanitizer** - Memory error detection

### Community
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)
- [Stack Overflow - CUDA Tag](https://stackoverflow.com/questions/tagged/cuda)
- [CUDA by Example (Book)](http://www.cudabyexample.com/)

---

## Statistics

- **Total Programs**: 67 .cu files + 56 notebooks = 123 learning resources
- **Lines of Code**: ~21,000 lines of CUDA code
- **Concepts Covered**: 80+ CUDA programming concepts
- **Learning Time**: 8-11 weeks for complete mastery (1-2 hours/day)
- **Cost**: Free with Google Colab

---

## Ready to Start?

### For Local Development:
```bash
cd local/phase1
nvcc -arch=sm_70 01_hello_world.cu -o hello
./hello
```

### For Google Colab:
1. Open https://colab.research.google.com
2. Upload `colab/notebooks/phase1/00-setup-verification.ipynb`
3. Runtime → Change runtime type → GPU (T4)
4. Run all cells and start learning!

---

**🚀 Your CUDA journey starts now! Happy GPU programming!**
