# CUDA Learning Curriculum - Jupyter Notebooks

This directory contains 55 comprehensive Jupyter notebooks covering the complete CUDA programming curriculum from beginner to expert level.

## Quick Start

1. Open any notebook in Google Colab
2. Ensure GPU runtime is enabled: Runtime → Change runtime type → GPU
3. Run cells sequentially
4. Complete exercises in each notebook

## Curriculum Structure

### Phase 1: Foundations (Notebooks 01-05)
**Duration:** Week 1-2
**Goal:** Master CUDA basics and kernel programming

- **01_hello_world.ipynb** - First CUDA kernel, basic syntax
- **02_device_query.ipynb** - GPU architecture and properties
- **03_vector_add.ipynb** - Memory management and vector operations
- **04_matrix_add.ipynb** - 2D thread organization and matrices
- **05_thread_indexing.ipynb** - Advanced indexing and grid-stride loops

**Milestone:** Can write basic kernels and manage GPU memory

---

### Phase 2: Memory Management (Notebooks 06-11)
**Duration:** Week 3-4
**Goal:** Master memory hierarchy and optimization

- **06_memory_basics.ipynb** - Memory allocation and data transfer
- **07_bandwidth_test.ipynb** - Memory bandwidth benchmarking
- **08_unified_memory.ipynb** - Unified Memory and managed memory
- **09_shared_memory_basics.ipynb** - Shared memory fundamentals
- **10_matrix_multiply_tiled.ipynb** - Tiled matrix multiplication
- **11_coalescing_demo.ipynb** - Memory coalescing patterns

**Milestone:** Can optimize memory access patterns for performance

---

### Phase 3: Optimization Fundamentals (Notebooks 12-17)
**Duration:** Week 5-6
**Goal:** Master warp-level programming and parallel algorithms

- **12_warp_divergence.ipynb** - Understanding and avoiding divergence
- **13_warp_shuffle.ipynb** - Warp-level primitives (__shfl)
- **14_occupancy_tuning.ipynb** - Optimizing GPU occupancy
- **15_parallel_reduction.ipynb** - Efficient parallel reduction
- **16_prefix_sum.ipynb** - Scan algorithms (inclusive/exclusive)
- **17_histogram.ipynb** - Histogram with atomic operations

**Milestone:** Can implement and optimize parallel algorithms

---

### Phase 4: Advanced Memory & Synchronization (Notebooks 18-23)
**Duration:** Week 7-8
**Goal:** Master advanced memory types and synchronization

- **18_texture_memory.ipynb** - Texture memory for image processing
- **19_constant_memory.ipynb** - Using constant memory
- **20_zero_copy.ipynb** - Zero-copy memory access
- **21_atomics.ipynb** - Atomic operation patterns
- **22_cooperative_groups.ipynb** - Cooperative groups API
- **23_multi_kernel_sync.ipynb** - Kernel synchronization

**Milestone:** Can use all memory types and synchronization primitives

---

### Phase 5: Advanced Algorithms (Notebooks 24-29)
**Duration:** Week 9-10
**Goal:** Implement complex algorithms and use libraries

- **24_gemm_optimized.ipynb** - Highly optimized matrix multiply
- **25_cublas_integration.ipynb** - Using cuBLAS library
- **26_matrix_transpose.ipynb** - Efficient transpose algorithm
- **27_bitonic_sort.ipynb** - Parallel bitonic sort
- **28_radix_sort.ipynb** - Radix sort implementation
- **29_thrust_examples.ipynb** - Thrust library usage

**Milestone:** Can implement production-quality algorithms

---

### Phase 6: Streams & Concurrency (Notebooks 30-35)
**Duration:** Week 11
**Goal:** Master asynchronous execution and multi-GPU

- **30_streams_basic.ipynb** - CUDA streams basics
- **31_async_pipeline.ipynb** - Asynchronous pipeline
- **32_events_timing.ipynb** - Events and performance timing
- **33_multi_gpu_basic.ipynb** - Multi-GPU programming
- **34_p2p_transfer.ipynb** - Peer-to-peer transfers
- **35_nccl_collectives.ipynb** - NCCL for multi-GPU

**Milestone:** Can overlap operations and use multiple GPUs

---

### Phase 7: Performance Engineering (Notebooks 36-40)
**Duration:** Week 12-13
**Goal:** Master profiling, debugging, and optimization

- **36_profiling_demo.ipynb** - Nsight Compute profiling
- **37_debugging_cuda.ipynb** - Debugging techniques
- **38_kernel_fusion.ipynb** - Kernel fusion optimization
- **39_fast_math.ipynb** - Fast math operations
- **40_advanced_optimization.ipynb** - Advanced techniques

**Milestone:** Can profile and optimize complex kernels

---

### Phase 8: Real-World Applications (Notebooks 41-49)
**Duration:** Week 14-15
**Goal:** Build complete GPU-accelerated applications

- **41_cufft_demo.ipynb** - cuFFT for Fourier transforms
- **42_cusparse_demo.ipynb** - cuSPARSE for sparse matrices
- **43_curand_demo.ipynb** - cuRAND for random numbers
- **44_image_processing.ipynb** - Complete image pipeline
- **45_raytracer.ipynb** - GPU ray tracing
- **46_nbody_simulation.ipynb** - N-body physics
- **47_neural_network.ipynb** - Neural network from scratch
- **48_molecular_dynamics.ipynb** - Molecular dynamics
- **49_option_pricing.ipynb** - Financial Monte Carlo

**Milestone:** Can build complete GPU applications

---

### Phase 9: Advanced Topics (Notebooks 50-55)
**Duration:** Week 16+
**Goal:** Master cutting-edge CUDA features

- **50_dynamic_parallelism.ipynb** - Dynamic parallelism
- **51_cuda_graphs.ipynb** - CUDA graphs
- **52_mps_demo.ipynb** - Multi-Process Service
- **53_mixed_precision.ipynb** - Mixed precision computing
- **54_tensor_cores.ipynb** - Tensor core programming
- **55_wmma_gemm.ipynb** - WMMA matrix multiply

**Milestone:** CUDA expert level achieved!

---

## Learning Path

### Beginner Path (4-6 weeks)
1. Complete Phase 1 (notebooks 01-05)
2. Complete Phase 2 (notebooks 06-11)
3. Practice exercises in each notebook
4. Build simple CUDA applications

### Intermediate Path (8-10 weeks)
1. Complete Beginner Path
2. Complete Phase 3 (notebooks 12-17)
3. Complete Phase 4 (notebooks 18-23)
4. Complete Phase 5 (notebooks 24-29)
5. Implement parallel algorithms from scratch

### Advanced Path (12-16 weeks)
1. Complete Intermediate Path
2. Complete Phase 6 (notebooks 30-35)
3. Complete Phase 7 (notebooks 36-40)
4. Complete Phase 8 (notebooks 41-49)
5. Build complete GPU-accelerated projects

### Expert Path (16+ weeks)
1. Complete Advanced Path
2. Complete Phase 9 (notebooks 50-55)
3. Contribute to open-source GPU projects
4. Optimize existing GPU libraries

---

## Notebook Features

Each notebook includes:

1. **Title and Phase Information** - Context within curriculum
2. **Learning Objectives** - 3-5 specific goals
3. **Concept Explanation** - Detailed theory with diagrams
4. **Multiple Code Examples** - Progressive complexity
5. **CUDA Code Cells** - Using %%cu magic for Colab
6. **Practical Exercises** - Hands-on practice
7. **Key Takeaways** - Summary of important points
8. **Next Steps** - Link to next notebook
9. **Notes Section** - Personal learning notes

---

## Prerequisites

- Strong C/C++ programming skills
- Understanding of pointers and memory management
- Basic understanding of parallel programming concepts
- Google Colab account (free) or local CUDA installation

---

## Hardware Requirements

### Minimum:
- Any CUDA-capable NVIDIA GPU (Compute Capability 3.0+)
- Google Colab Free Tier (T4 GPU)

### Recommended:
- NVIDIA GPU with Compute Capability 7.0+ (Volta, Turing, Ampere, Ada)
- 8GB+ GPU memory
- Google Colab Pro (A100 or V100 GPU)

### For Advanced Notebooks:
- Compute Capability 8.0+ for Tensor Core examples
- Multi-GPU system for Phase 6 notebooks
- 16GB+ GPU memory for large-scale examples

---

## How to Use These Notebooks

### In Google Colab:

1. **Upload to Google Drive:**
   ```bash
   # Upload entire notebooks folder to your Google Drive
   ```

2. **Open in Colab:**
   - Navigate to notebook in Drive
   - Right-click → Open with → Google Colaboratory

3. **Enable GPU:**
   - Runtime → Change runtime type
   - Hardware accelerator → GPU
   - GPU type → T4 (free) or A100 (Pro)

4. **Run Notebooks:**
   - Run cells sequentially
   - Complete exercises
   - Experiment with parameters

### Local Installation:

1. **Install CUDA Toolkit:**
   ```bash
   # Download from NVIDIA website
   # Install CUDA 11.8 or later
   ```

2. **Install Jupyter:**
   ```bash
   pip install jupyter
   pip install nvcc4jupyter
   ```

3. **Load CUDA Extension:**
   ```python
   %load_ext nvcc4jupyter
   ```

4. **Run Notebooks:**
   ```bash
   jupyter notebook
   ```

---

## Learning Tips

1. **Follow Sequential Order:** Notebooks build on previous concepts
2. **Complete All Exercises:** Practice is essential
3. **Experiment:** Modify code and observe results
4. **Profile Everything:** Measure performance improvements
5. **Take Notes:** Use the Notes section in each notebook
6. **Compare with Libraries:** Understand how production code works
7. **Read CUDA Documentation:** Official docs are excellent
8. **Join Communities:** NVIDIA Developer Forums, Reddit r/CUDA

---

## Common Issues and Solutions

### Issue: CUDA out of memory
**Solution:** Reduce data size, use smaller batch sizes, or restart runtime

### Issue: Kernel launch failures
**Solution:** Check grid/block dimensions, enable error checking

### Issue: Slow performance
**Solution:** Profile with Nsight, check memory coalescing, reduce divergence

### Issue: Incorrect results
**Solution:** Add synchronization, check boundary conditions, use cuda-memcheck

---

## Performance Metrics

Track your progress with these benchmarks:

### Phase 1-2: Vector Addition
- Target: 100+ GB/s memory bandwidth

### Phase 2-3: Matrix Multiplication
- Target: 1000+ GFLOPS on 1024x1024

### Phase 3: Parallel Reduction
- Target: 200+ GB/s effective bandwidth

### Phase 5: Optimized GEMM
- Target: 50%+ of cuBLAS performance

### Phase 8: Real Applications
- Target: 10x+ speedup over CPU

---

## Additional Resources

### Books:
- "Programming Massively Parallel Processors" by Kirk & Hwu
- "CUDA by Example" by Sanders & Kandrot
- "Professional CUDA C Programming" by Cheng et al.

### Online:
- NVIDIA CUDA C Programming Guide
- NVIDIA CUDA Best Practices Guide
- CUDA Samples on GitHub

### Tools:
- Nsight Compute (kernel profiler)
- Nsight Systems (system profiler)
- cuda-gdb (debugger)
- Compute Sanitizer (memory checker)

### Courses:
- NVIDIA Deep Learning Institute
- Coursera GPU Programming
- Udacity Intro to Parallel Programming

---

## Contributing

Found an issue or want to improve a notebook?
1. Fork the repository
2. Make improvements
3. Submit a pull request

---

## License

These educational materials are provided for learning purposes.

---

## Acknowledgments

Based on the CUDA Learning Curriculum designed for developers transitioning from C/C++ to CUDA programming.

---

**Ready to start?** Open **01_hello_world.ipynb** and begin your CUDA journey!

**Questions?** Check the curriculum document or NVIDIA documentation.

**Happy CUDA Programming!** 🚀
