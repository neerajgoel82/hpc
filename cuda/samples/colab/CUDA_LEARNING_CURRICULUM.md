# CUDA Programming Curriculum
## From C/C++ Developer to CUDA Expert

This curriculum is designed for developers with strong C/C++ skills who want to master CUDA programming.

---

## Phase 1: Foundations (Week 1-2)

### Module 1.1: CUDA Architecture Basics
**Concepts to Learn:**
- GPU vs CPU architecture differences
- CUDA programming model overview
- Host (CPU) vs Device (GPU) code
- NVIDIA GPU architecture: SMs, cores, warps, threads
- Memory hierarchy: registers, shared memory, global memory

**Practical Exercise:**
- Write your first "Hello World" from GPU
- Query GPU device properties
- Understand compilation with nvcc

**Sample Projects:**
- `01-hello-world/` - Basic kernel launch
- `02-device-query/` - Enumerate GPU capabilities

---

### Module 1.2: Thread Hierarchy & Kernel Basics
**Concepts to Learn:**
- Thread, block, and grid organization
- Block dimensions (1D, 2D, 3D)
- Grid dimensions (1D, 2D, 3D)
- Built-in variables: `threadIdx`, `blockIdx`, `blockDim`, `gridDim`
- Calculating global thread ID

**Practical Exercise:**
- Vector addition on GPU
- Matrix operations with 2D thread blocks
- Understanding kernel launch syntax `<<<grid, block>>>`

**Sample Projects:**
- `03-vector-add/` - Simple parallel vector addition
- `04-matrix-add/` - 2D block/grid matrix addition
- `05-thread-indexing/` - Practice calculating global indices

---

## Phase 2: Memory Management (Week 3-4)

### Module 2.1: Memory Types & Data Transfer
**Concepts to Learn:**
- Global memory allocation (`cudaMalloc`, `cudaFree`)
- Data transfer (`cudaMemcpy`)
- Host-device synchronization
- Pinned (page-locked) memory
- Unified Memory basics

**Practical Exercise:**
- Measure memory transfer bandwidth
- Compare pinned vs pageable memory performance
- Use unified memory for simpler code

**Sample Projects:**
- `06-memory-basics/` - Allocation and transfer patterns
- `07-bandwidth-test/` - Memory bandwidth benchmarking
- `08-unified-memory/` - Using managed memory

---

### Module 2.2: Shared Memory & Memory Coalescing
**Concepts to Learn:**
- Shared memory: on-chip, fast, limited
- Memory coalescing for global memory access
- Bank conflicts in shared memory
- Proper memory access patterns
- Memory alignment

**Practical Exercise:**
- Implement tiled matrix multiplication
- Optimize global memory access patterns
- Avoid/resolve bank conflicts

**Sample Projects:**
- `09-shared-memory-basics/` - Introduction to shared memory
- `10-matrix-multiply-tiled/` - Classic tiled matmul
- `11-coalescing-demo/` - Memory access pattern optimization

---

## Phase 3: Optimization Fundamentals (Week 5-6)

### Module 3.1: Warp-Level Programming
**Concepts to Learn:**
- Warp execution model (32 threads)
- Warp divergence and branching
- Warp-level primitives (`__shfl`, `__ballot`, etc.)
- Occupancy and resource usage
- Register pressure

**Practical Exercise:**
- Analyze and reduce warp divergence
- Use warp shuffle for communication
- Profile with nvprof/Nsight Compute

**Sample Projects:**
- `12-warp-divergence/` - Demonstrating divergence impact
- `13-warp-shuffle/` - Using warp-level primitives
- `14-occupancy-tuning/` - Optimize for occupancy

---

### Module 3.2: Reduction & Scan Patterns
**Concepts to Learn:**
- Parallel reduction algorithms
- Tree-based reduction
- Warp-level reduction
- Prefix sum (scan) algorithms
- Sequential addressing vs interleaved

**Practical Exercise:**
- Implement efficient parallel reduction
- Optimize reduction with shared memory
- Implement inclusive/exclusive scan

**Sample Projects:**
- `15-parallel-reduction/` - Multi-stage reduction optimization
- `16-prefix-sum/` - Scan implementation
- `17-histogram/` - Using atomic operations

---

## Phase 4: Advanced Memory & Synchronization (Week 7-8)

### Module 4.1: Advanced Memory Techniques
**Concepts to Learn:**
- Texture memory and caching
- Constant memory
- Read-only data cache (LDG)
- Memory pooling and reuse
- Zero-copy memory

**Practical Exercise:**
- Use texture memory for image processing
- Optimize with constant memory
- Implement memory pool

**Sample Projects:**
- `18-texture-memory/` - Image filtering with textures
- `19-constant-memory/` - Using const memory for coefficients
- `20-zero-copy/` - Direct host memory access

---

### Module 4.2: Synchronization Patterns
**Concepts to Learn:**
- `__syncthreads()` and barriers
- Atomic operations (atomicAdd, atomicCAS, etc.)
- Memory fences
- Cooperative groups
- Grid-level synchronization

**Practical Exercise:**
- Implement thread-safe algorithms
- Use cooperative groups for flexible synchronization
- Understand memory ordering

**Sample Projects:**
- `21-atomics/` - Atomic operation patterns
- `22-cooperative-groups/` - Using cooperative groups API
- `23-multi-kernel-sync/` - Kernel dependencies

---

## Phase 5: Advanced Algorithms (Week 9-10)

### Module 5.1: Matrix Operations & Linear Algebra
**Concepts to Learn:**
- Optimized matrix multiplication (beyond tiled)
- Using cuBLAS library
- Tensor cores (if available)
- Matrix transpose optimization
- Block algorithms

**Practical Exercise:**
- Implement highly-optimized GEMM
- Use cuBLAS for production code
- Optimize transpose with shared memory

**Sample Projects:**
- `24-gemm-optimized/` - Advanced matrix multiply
- `25-cublas-integration/` - Using cuBLAS library
- `26-matrix-transpose/` - Efficient transpose

---

### Module 5.2: Sorting & Search Algorithms
**Concepts to Learn:**
- Parallel sorting (bitonic, radix, merge)
- Binary search on GPU
- Using Thrust library
- CUB (CUDA Unbound) primitives
- Multi-GPU sorting

**Practical Exercise:**
- Implement radix sort
- Use Thrust for productivity
- Compare GPU vs CPU sorting

**Sample Projects:**
- `27-bitonic-sort/` - Parallel bitonic sort
- `28-radix-sort/` - Efficient radix sort
- `29-thrust-examples/` - Using Thrust library

---

## Phase 6: Streams & Concurrency (Week 11)

### Module 6.1: Streams & Asynchronous Execution
**Concepts to Learn:**
- CUDA streams concept
- Overlapping compute and memory transfer
- Stream priorities
- Events and timing
- Callback functions

**Practical Exercise:**
- Pipeline with multiple streams
- Measure performance gains
- Use events for synchronization

**Sample Projects:**
- `30-streams-basic/` - Introduction to streams
- `31-async-pipeline/` - Overlapping operations
- `32-events-timing/` - Performance measurement

---

### Module 6.2: Multi-GPU Programming
**Concepts to Learn:**
- Multiple GPU management
- Peer-to-peer memory access
- Load balancing across GPUs
- NCCL for multi-GPU communication
- GPU Direct technologies

**Practical Exercise:**
- Distribute work across multiple GPUs
- Implement peer-to-peer transfer
- Compare scaling efficiency

**Sample Projects:**
- `33-multi-gpu-basic/` - Managing multiple devices
- `34-p2p-transfer/` - Peer-to-peer memory copy
- `35-nccl-collectives/` - Multi-GPU communication

---

## Phase 7: Performance Engineering (Week 12-13)

### Module 7.1: Profiling & Debugging
**Concepts to Learn:**
- Using Nsight Compute for kernel analysis
- Using Nsight Systems for timeline view
- cuda-memcheck for memory errors
- compute-sanitizer
- Performance metrics (bandwidth, occupancy, IPC)

**Practical Exercise:**
- Profile real applications
- Identify bottlenecks
- Debug memory issues

**Sample Projects:**
- `36-profiling-demo/` - What to measure and how
- `37-debugging-cuda/` - Common bugs and fixes

---

### Module 7.2: Advanced Optimization Techniques
**Concepts to Learn:**
- Instruction-level parallelism
- Loop unrolling
- Using PTX and inline assembly
- Fast math operations
- Kernel fusion
- Grid-stride loops

**Practical Exercise:**
- Manual optimization with PTX
- Benchmark fast math vs precise math
- Optimize complex kernels

**Sample Projects:**
- `38-kernel-fusion/` - Combining operations
- `39-fast-math/` - Trading precision for speed
- `40-advanced-optimization/` - Multiple techniques combined

---

## Phase 8: Real-World Applications (Week 14-15)

### Module 8.1: Domain-Specific Libraries
**Concepts to Learn:**
- cuFFT for Fourier transforms
- cuDNN for deep learning
- cuSPARSE for sparse matrices
- cuRAND for random numbers
- cuSOLVER for linear algebra

**Practical Exercise:**
- FFT-based convolution
- Use cuDNN for neural net layers
- Sparse matrix operations

**Sample Projects:**
- `41-cufft-demo/` - FFT applications
- `42-cusparse-demo/` - Sparse linear algebra
- `43-curand-demo/` - Monte Carlo simulation

---

### Module 8.2: Complete Projects
**Practical Projects:**

**Sample Projects:**
- `44-image-processing/` - Complete image processing pipeline
- `45-raytracer/` - GPU ray tracing
- `46-nbody-simulation/` - N-body physics simulation
- `47-neural-network/` - Simple neural network from scratch
- `48-molecular-dynamics/` - MD simulation
- `49-option-pricing/` - Financial Monte Carlo

---

## Phase 9: Advanced Topics (Week 16+)

### Module 9.1: Modern CUDA Features
**Concepts to Learn:**
- Dynamic parallelism
- CUDA graphs
- Graph capture and instantiation
- GPU Direct Storage
- Multi-Process Service (MPS)

**Sample Projects:**
- `50-dynamic-parallelism/` - Kernels launching kernels
- `51-cuda-graphs/` - Graph-based execution
- `52-mps-demo/` - Multi-process sharing

---

### Module 9.2: Mixed Precision & Tensor Cores
**Concepts to Learn:**
- FP16, BF16, TF32, INT8 compute
- Tensor core programming
- WMMA API (Warp Matrix Multiply-Accumulate)
- Mixed precision strategies

**Sample Projects:**
- `53-mixed-precision/` - Using multiple precisions
- `54-tensor-cores/` - Programming tensor cores
- `55-wmma-gemm/` - Matrix multiply with WMMA

---

## Recommended Resources

### Books
- "Programming Massively Parallel Processors" by Kirk & Hwu
- "CUDA by Example" by Sanders & Kandrot
- "Professional CUDA C Programming" by Cheng, Grossman & McKercher

### Online Resources
- NVIDIA CUDA C Programming Guide
- NVIDIA CUDA Best Practices Guide
- CUDA Samples on GitHub
- GPU Gems series (free online)

### Tools
- NVIDIA Nsight Compute (kernel profiler)
- NVIDIA Nsight Systems (system-wide timeline)
- cuda-gdb (debugger)
- Compute Sanitizer (memory checker)

### Practice Platforms
- NVIDIA DLI (Deep Learning Institute) courses
- Coursera/Udacity GPU programming courses
- LeetCode/HackerRank GPU problems

---

## Assessment Milestones

### Beginner (Phase 1-2)
- Can write basic kernels with proper thread indexing
- Understands memory transfer and management
- Can implement simple parallel algorithms

### Intermediate (Phase 3-5)
- Optimizes code using shared memory
- Understands warp-level programming
- Can implement complex algorithms (reduction, sort, matmul)
- Uses profiling tools effectively

### Advanced (Phase 6-7)
- Uses streams and multi-GPU programming
- Can identify and fix performance bottlenecks
- Writes production-quality optimized code

### Expert (Phase 8-9)
- Builds complete GPU-accelerated applications
- Uses domain-specific libraries effectively
- Understands cutting-edge CUDA features
- Can make architectural decisions for GPU computing

---

## Daily Practice Recommendations

1. **Code daily**: Even 30 minutes of practice helps
2. **Profile everything**: Always measure performance
3. **Read CUDA samples**: NVIDIA's official samples are excellent
4. **Experiment**: Try different optimization strategies
5. **Benchmark**: Compare your implementations with libraries
6. **Debug carefully**: Use proper tools (cuda-memcheck, compute-sanitizer)
7. **Read documentation**: CUDA docs are comprehensive and well-written

---

## Project Structure for This Repository

```
cuda-samples/
├── 01-hello-world/
├── 02-device-query/
├── 03-vector-add/
├── ...
├── 55-wmma-gemm/
├── common/           # Shared utilities
│   ├── helper_cuda.h
│   ├── timer.h
│   └── utils.h
└── CUDA_LEARNING_CURRICULUM.md (this file)
```

Each project directory should contain:
- `*.cu` source files
- `Makefile` or `CMakeLists.txt`
- `README.md` with explanation and learning objectives
- Reference solutions and optimization notes

---

## Getting Started

1. Ensure you have CUDA Toolkit installed
2. Verify installation: `nvcc --version`
3. Start with Phase 1, Module 1.1
4. Create each project directory as you progress
5. Commit working code to track your learning journey

Happy CUDA programming!
