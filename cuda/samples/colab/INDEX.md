# CUDA Learning Curriculum - Complete Index

## 🎯 Quick Navigation

- **Total Notebooks:** 55 curriculum + 1 setup = 56 total
- **Status:** ✅ Complete
- **Format:** Jupyter Notebooks (.ipynb)
- **Platform:** Google Colab compatible
- **Start Here:** [notebooks/phase1/01_hello_world.ipynb](notebooks/phase1/01_hello_world.ipynb)

---

## 📚 Complete Notebook Index

### 🔰 Phase 1: Foundations (Week 1-2)
*Master CUDA basics and kernel programming*

| # | Notebook | Description | Key Topics |
|---|----------|-------------|------------|
| 01 | [Hello World](notebooks/phase1/01_hello_world.ipynb) | Your first CUDA kernel | Kernels, thread launch, synchronization |
| 02 | [Device Query](notebooks/phase1/02_device_query.ipynb) | GPU architecture exploration | Device properties, compute capability, SM count |
| 03 | [Vector Add](notebooks/phase1/03_vector_add.ipynb) | Parallel vector operations | Memory allocation, data transfer, verification |
| 04 | [Matrix Add](notebooks/phase1/04_matrix_add.ipynb) | 2D thread organization | 2D grids, dim3, matrix operations |
| 05 | [Thread Indexing](notebooks/phase1/05_thread_indexing.ipynb) | Advanced indexing patterns | Grid-stride loops, 3D indexing, utilities |

**Completion Goal:** Can write basic kernels and manage GPU memory

---

### 💾 Phase 2: Memory Management (Week 3-4)
*Master memory hierarchy and optimization*

| # | Notebook | Description | Key Topics |
|---|----------|-------------|------------|
| 06 | [Memory Basics](notebooks/phase2/06_memory_basics_and_data_transfer.ipynb) | Memory allocation fundamentals | cudaMalloc, cudaMemcpy, pinned memory |
| 07 | [Bandwidth Test](notebooks/phase2/07_memory_bandwidth_benchmarking.ipynb) | Measuring memory performance | PCIe bandwidth, transfer timing |
| 08 | [Unified Memory](notebooks/phase2/08_unified_memory_and_managed_memory.ipynb) | Managed memory usage | cudaMallocManaged, prefetching |
| 09 | [Shared Memory](notebooks/phase2/09_shared_memory_basics.ipynb) | On-chip fast memory | __shared__, __syncthreads(), tiling |
| 10 | [Tiled MatMul](notebooks/phase2/10_tiled_matrix_multiplication.ipynb) | Optimized matrix multiply | Blocking, shared memory optimization |
| 11 | [Coalescing](notebooks/phase2/11_memory_coalescing_demonstration.ipynb) | Memory access patterns | Coalesced access, stride patterns |

**Completion Goal:** Can optimize memory access for performance

---

### ⚡ Phase 3: Optimization Fundamentals (Week 5-6)
*Master warp-level programming*

| # | Notebook | Description | Key Topics |
|---|----------|-------------|------------|
| 12 | [Warp Divergence](notebooks/phase3/12_warp_divergence.ipynb) | Understanding warp execution | Branch efficiency, divergence impact |
| 13 | [Warp Shuffle](notebooks/phase3/13_warp_shuffle.ipynb) | Warp-level communication | __shfl, __ballot, warp primitives |
| 14 | [Occupancy Tuning](notebooks/phase3/14_occupancy_tuning.ipynb) | Resource optimization | Occupancy calculation, register pressure |
| 15 | [Parallel Reduction](notebooks/phase3/15_parallel_reduction.ipynb) | Sum/min/max algorithms | Tree reduction, shared memory reduction |
| 16 | [Prefix Sum](notebooks/phase3/16_prefix_sum.ipynb) | Scan operations | Inclusive/exclusive scan, work-efficient |
| 17 | [Histogram](notebooks/phase3/17_histogram.ipynb) | Binning with atomics | Atomic operations, privatization |

**Completion Goal:** Can implement efficient parallel algorithms

---

### 🔬 Phase 4: Advanced Memory & Synchronization (Week 7-8)
*Master advanced memory types*

| # | Notebook | Description | Key Topics |
|---|----------|-------------|------------|
| 18 | [Texture Memory](notebooks/phase4/18_texture_memory.ipynb) | Texture cache usage | Texture objects, filtering |
| 19 | [Constant Memory](notebooks/phase4/19_constant_memory.ipynb) | Read-only cached data | Constant cache, coefficients |
| 20 | [Zero-Copy](notebooks/phase4/20_zero_copy.ipynb) | Direct host access | Mapped memory, PCIe access |
| 21 | [Atomics](notebooks/phase4/21_atomics.ipynb) | Thread-safe operations | atomicAdd, atomicCAS, patterns |
| 22 | [Cooperative Groups](notebooks/phase4/22_cooperative_groups.ipynb) | Flexible synchronization | Thread groups, grid sync |
| 23 | [Multi-Kernel Sync](notebooks/phase4/23_multi_kernel_sync.ipynb) | Kernel dependencies | Streams, events, dependencies |

**Completion Goal:** Can use all memory types effectively

---

### 🚀 Phase 5: Advanced Algorithms (Week 9-10)
*Implement production algorithms*

| # | Notebook | Description | Key Topics |
|---|----------|-------------|------------|
| 24 | [Optimized GEMM](notebooks/phase5/24_gemm_optimized.ipynb) | Matrix multiply expert | Register tiling, optimization |
| 25 | [cuBLAS](notebooks/phase5/25_cublas_integration.ipynb) | Using cuBLAS library | Library integration, BLAS ops |
| 26 | [Matrix Transpose](notebooks/phase5/26_matrix_transpose.ipynb) | Efficient transpose | Bank conflict avoidance |
| 27 | [Bitonic Sort](notebooks/phase5/27_bitonic_sort.ipynb) | Comparison network sort | Parallel sorting, bitonic merge |
| 28 | [Radix Sort](notebooks/phase5/28_radix_sort.ipynb) | Digit-based sorting | Histogram, prefix sum, reorder |
| 29 | [Thrust Library](notebooks/phase5/29_thrust_examples.ipynb) | STL-like GPU algorithms | Thrust vectors, algorithms |

**Completion Goal:** Can implement complex algorithms

---

### 🔀 Phase 6: Streams & Concurrency (Week 11)
*Master asynchronous execution*

| # | Notebook | Description | Key Topics |
|---|----------|-------------|------------|
| 30 | [CUDA Streams](notebooks/phase6/30_streams_basic.ipynb) | Stream fundamentals | Stream creation, async ops |
| 31 | [Async Pipeline](notebooks/phase6/31_async_pipeline.ipynb) | Overlapping operations | Compute-transfer overlap |
| 32 | [Events & Timing](notebooks/phase6/32_events_timing.ipynb) | Performance measurement | cudaEvent, timing, profiling |
| 33 | [Multi-GPU Basic](notebooks/phase6/33_multi_gpu_basic.ipynb) | Multiple GPU usage | Device management, workload split |
| 34 | [P2P Transfer](notebooks/phase6/34_p2p_transfer.ipynb) | Inter-GPU communication | Peer-to-peer, GPU Direct |
| 35 | [NCCL](notebooks/phase6/35_nccl_collectives.ipynb) | Multi-GPU collectives | NCCL library, allreduce |

**Completion Goal:** Can use multiple GPUs efficiently

---

### 🔧 Phase 7: Performance Engineering (Week 12-13)
*Master profiling and optimization*

| # | Notebook | Description | Key Topics |
|---|----------|-------------|------------|
| 36 | [Profiling](notebooks/phase7/36_profiling_demo.ipynb) | Nsight tools usage | Kernel profiling, metrics |
| 37 | [Debugging](notebooks/phase7/37_debugging_cuda.ipynb) | Finding and fixing bugs | cuda-memcheck, sanitizer |
| 38 | [Kernel Fusion](notebooks/phase7/38_kernel_fusion.ipynb) | Combining operations | Launch overhead reduction |
| 39 | [Fast Math](notebooks/phase7/39_fast_math.ipynb) | Fast intrinsics | Precision tradeoffs, speed |
| 40 | [Advanced Opts](notebooks/phase7/40_advanced_optimization.ipynb) | Expert techniques | ILP, loop unrolling, PTX |

**Completion Goal:** Can optimize production code

---

### 🌟 Phase 8: Real-World Applications (Week 14-15)
*Build complete applications*

| # | Notebook | Description | Key Topics |
|---|----------|-------------|------------|
| 41 | [cuFFT](notebooks/phase8/41_cufft_demo.ipynb) | Fourier transforms | FFT, frequency domain |
| 42 | [cuSPARSE](notebooks/phase8/42_cusparse_demo.ipynb) | Sparse linear algebra | Sparse matrices, CSR |
| 43 | [cuRAND](notebooks/phase8/43_curand_demo.ipynb) | Random number generation | RNG, Monte Carlo |
| 44 | [Image Processing](notebooks/phase8/44_image_processing.ipynb) | Complete pipeline | Filters, convolution |
| 45 | [Ray Tracer](notebooks/phase8/45_raytracer.ipynb) | GPU rendering | Ray tracing, graphics |
| 46 | [N-Body](notebooks/phase8/46_nbody_simulation.ipynb) | Physics simulation | Gravitational forces |
| 47 | [Neural Network](notebooks/phase8/47_neural_network.ipynb) | Deep learning basics | Layers, backprop |
| 48 | [Molecular Dynamics](notebooks/phase8/48_molecular_dynamics.ipynb) | MD simulation | Force fields, integration |
| 49 | [Option Pricing](notebooks/phase8/49_option_pricing.ipynb) | Financial computing | Monte Carlo, options |

**Completion Goal:** Can build complete GPU applications

---

### 🎓 Phase 9: Advanced Topics (Week 16+)
*Master cutting-edge features*

| # | Notebook | Description | Key Topics |
|---|----------|-------------|------------|
| 50 | [Dynamic Parallelism](notebooks/phase9/50_dynamic_parallelism.ipynb) | Kernels launch kernels | Nested parallelism, recursion |
| 51 | [CUDA Graphs](notebooks/phase9/51_cuda_graphs.ipynb) | Graph-based execution | Graph capture, optimization |
| 52 | [MPS](notebooks/phase9/52_mps_demo.ipynb) | Multi-process service | GPU sharing, MPS |
| 53 | [Mixed Precision](notebooks/phase9/53_mixed_precision.ipynb) | Multiple data types | FP16, FP32, accuracy |
| 54 | [Tensor Cores](notebooks/phase9/54_tensor_cores.ipynb) | Matrix accelerators | Tensor cores, AI ops |
| 55 | [WMMA](notebooks/phase9/55_wmma_gemm.ipynb) | Warp matrix ops | WMMA API, tensor cores |

**Completion Goal:** CUDA Expert! 🏆

---

## 🎓 Learning Paths

### 🐣 Absolute Beginner (0-4 weeks)
**Goal:** Understand GPU programming basics

**Path:**
1. Phase 1: Notebooks 01-05
2. Simple exercises
3. CPU vs GPU comparisons

**Time:** 20-30 hours

---

### 🔰 Beginner (4-8 weeks)
**Goal:** Write optimized CUDA kernels

**Path:**
1. Complete Phase 1
2. Complete Phase 2
3. Implement memory optimizations

**Time:** 40-60 hours

---

### 📊 Intermediate (8-12 weeks)
**Goal:** Implement complex algorithms

**Path:**
1. Complete Phases 1-2
2. Complete Phases 3-5
3. Build parallel algorithms

**Time:** 80-120 hours

---

### 🚀 Advanced (12-16 weeks)
**Goal:** Build production applications

**Path:**
1. Complete Phases 1-5
2. Complete Phases 6-8
3. Multi-GPU projects

**Time:** 120-160 hours

---

### 🏆 Expert (16+ weeks)
**Goal:** Master all CUDA features

**Path:**
1. Complete all phases
2. Contribute to GPU projects
3. Research and optimization

**Time:** 160+ hours

---

## 📊 Progress Tracking

### Checklist by Phase

- [ ] **Phase 1 Complete** (5 notebooks)
- [ ] **Phase 2 Complete** (6 notebooks)
- [ ] **Phase 3 Complete** (6 notebooks)
- [ ] **Phase 4 Complete** (6 notebooks)
- [ ] **Phase 5 Complete** (6 notebooks)
- [ ] **Phase 6 Complete** (6 notebooks)
- [ ] **Phase 7 Complete** (5 notebooks)
- [ ] **Phase 8 Complete** (9 notebooks)
- [ ] **Phase 9 Complete** (6 notebooks)

### Skill Milestones

- [ ] Can launch CUDA kernels
- [ ] Can manage GPU memory
- [ ] Can optimize memory access
- [ ] Can use shared memory
- [ ] Can implement reductions
- [ ] Can use CUDA libraries
- [ ] Can use multiple GPUs
- [ ] Can profile and optimize
- [ ] Can build complete apps
- [ ] Master of CUDA! 🏆

---

## 🔗 Quick Links

- **Main README:** [notebooks/README.md](notebooks/README.md)
- **Curriculum:** [CUDA_LEARNING_CURRICULUM.md](CUDA_LEARNING_CURRICULUM.md)
- **Summary:** [NOTEBOOKS_SUMMARY.md](NOTEBOOKS_SUMMARY.md)
- **Verification:** [verify_notebooks.sh](verify_notebooks.sh)

---

## 📦 Download and Setup

### Google Colab (Recommended)
```bash
# 1. Upload to Google Drive
# 2. Open any .ipynb file
# 3. Right-click → Open with → Google Colaboratory
# 4. Runtime → Change runtime type → GPU
```

### Local Jupyter
```bash
# Install CUDA Toolkit
wget https://developer.nvidia.com/cuda-downloads

# Install Jupyter
pip install jupyter nvcc4jupyter

# Load extension
%load_ext nvcc4jupyter

# Start Jupyter
jupyter notebook
```

---

## 🎯 Daily Practice Schedule

### Week 1-2: Foundations
- **Day 1-2:** Notebooks 01-02 (CUDA basics)
- **Day 3-4:** Notebook 03 (Memory management)
- **Day 5-6:** Notebook 04 (2D operations)
- **Day 7:** Notebook 05 (Advanced indexing)

### Week 3-4: Memory
- **Day 1-2:** Notebooks 06-07 (Memory fundamentals)
- **Day 3-4:** Notebooks 08-09 (Unified & shared memory)
- **Day 5-7:** Notebooks 10-11 (Optimization)

### Week 5-6: Optimization
- **Day 1-2:** Notebooks 12-13 (Warp programming)
- **Day 3-4:** Notebook 14 (Occupancy)
- **Day 5-7:** Notebooks 15-17 (Algorithms)

### Week 7-8: Advanced Memory
- **Day 1-3:** Notebooks 18-20 (Memory types)
- **Day 4-7:** Notebooks 21-23 (Synchronization)

### Week 9-10: Algorithms
- **Day 1-3:** Notebooks 24-26 (Linear algebra)
- **Day 4-7:** Notebooks 27-29 (Sorting)

### Week 11: Concurrency
- **Day 1-2:** Notebooks 30-31 (Streams)
- **Day 3-5:** Notebooks 32-34 (Multi-GPU)
- **Day 6-7:** Notebook 35 (NCCL)

### Week 12-13: Performance
- **Day 1-3:** Notebooks 36-37 (Profile & debug)
- **Day 4-7:** Notebooks 38-40 (Optimize)

### Week 14-15: Applications
- **Day 1-2:** Notebooks 41-43 (Libraries)
- **Day 3-10:** Notebooks 44-49 (Projects)

### Week 16+: Advanced
- **Day 1-4:** Notebooks 50-52 (Modern CUDA)
- **Day 5-7:** Notebooks 53-55 (Tensor cores)

---

## 🏅 Certification Path

After completing all notebooks, you'll have mastered:

1. ✅ CUDA Programming Fundamentals
2. ✅ Memory Hierarchy Optimization
3. ✅ Parallel Algorithm Design
4. ✅ Multi-GPU Programming
5. ✅ Performance Engineering
6. ✅ Real-World Application Development
7. ✅ Advanced CUDA Features

**Next Steps:**
- Contribute to open-source GPU projects
- Take NVIDIA DLI courses
- Apply for CUDA certifications
- Build your own GPU library

---

## 📞 Support and Community

### Questions?
- Check notebook explanations
- Review CUDA documentation
- Ask in NVIDIA forums

### Issues?
- Verify GPU setup
- Check CUDA version
- Review error messages

### Want More?
- Explore CUDA samples
- Read research papers
- Join CUDA communities

---

**🚀 Ready to Begin?**

Start your CUDA journey now: [notebooks/phase1/01_hello_world.ipynb](notebooks/phase1/01_hello_world.ipynb)

**Happy CUDA Programming!**
