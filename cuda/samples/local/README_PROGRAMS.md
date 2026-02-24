# Local CUDA Programs - Complete Collection

Standalone `.cu` files that can be compiled and run with `nvcc` on systems with NVIDIA GPUs.

## 📁 Directory Structure

```
local/
├── phase1/          # Foundations (3 programs)
├── phase2/          # Memory Management (3 programs)
├── phase3/          # Optimization (1 program)
├── phase4/          # Advanced Memory (1 program)
├── phase5/          # Advanced Algorithms (1 program)
├── phase6/          # Streams & Concurrency (1 program)
├── phase7/          # Performance Engineering (1 program)
├── phase8/          # Real Applications (1 program)
├── phase9/          # Modern CUDA Features (1 program)
└── Makefile         # Root makefile for all phases
```

**Total Programs**: 13 complete, runnable CUDA programs

---

## 🚀 Quick Start

### Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed (11.0+)
- `nvcc` compiler in PATH

### Build All Programs

```bash
cd local
make                # Build all programs in all phases
```

### Build Specific Phase

```bash
cd local/phase1
make                # Build phase 1 programs only
```

### Run Programs

```bash
# Run all tests
cd local
make test

# Run specific program
cd local/phase1
./01_hello_world
./02_vector_add
./03_matrix_add
```

### Clean Build Artifacts

```bash
cd local
make clean          # Clean all phases
```

---

## 📋 Programs by Phase

### Phase 1: Foundations
**Focus**: Basic CUDA concepts and kernel launches

- `01_hello_world.cu` - First CUDA program with device query
- `02_vector_add.cu` - Vector addition with CPU vs GPU benchmark
- `03_matrix_add.cu` - 2D matrix addition with 2D grids

**Build**: `cd phase1 && make`

---

### Phase 2: Memory Management
**Focus**: Memory transfer and optimization

- `01_memory_bandwidth.cu` - Benchmark pageable vs pinned memory
- `02_shared_memory.cu` - Shared memory performance demonstration
- `03_coalescing.cu` - Memory coalescing impact on bandwidth

**Build**: `cd phase2 && make`

---

### Phase 3: Optimization
**Focus**: Parallel algorithms and optimization techniques

- `01_reduction.cu` - Optimized parallel reduction with CPU comparison

**Build**: `cd phase3 && make`

---

### Phase 4: Advanced Memory & Synchronization
**Focus**: Atomic operations and advanced memory types

- `01_atomics.cu` - Atomic histogram with performance timing

**Build**: `cd phase4 && make`

---

### Phase 5: Advanced Algorithms
**Focus**: Complex parallel algorithms

- `01_matmul_tiled.cu` - Tiled matrix multiplication with GFLOPS measurement

**Build**: `cd phase5 && make`

---

### Phase 6: Streams & Concurrency
**Focus**: Asynchronous execution and multi-stream programming

- `01_streams.cu` - Concurrent kernel execution with streams

**Build**: `cd phase6 && make`

---

### Phase 7: Performance Engineering
**Focus**: Performance tuning and optimization

- `01_occupancy.cu` - Occupancy tuning with different block sizes

**Build**: `cd phase7 && make`

---

### Phase 8: Real Applications
**Focus**: Production-style applications

- `01_nbody.cu` - N-body gravitational simulation

**Build**: `cd phase8 && make`

---

### Phase 9: Modern CUDA Features
**Focus**: Latest CUDA capabilities

- `01_cuda_graphs.cu` - CUDA Graphs for reduced launch overhead

**Build**: `cd phase9 && make`

---

## 🔧 Compilation Details

### Compiler Flags

All programs use:
```bash
nvcc -arch=sm_70 -O2 program.cu -o program
```

- `-arch=sm_70`: Volta architecture (adjust for your GPU)
- `-O2`: Optimization level 2

### Adjusting for Your GPU

Check your GPU's compute capability:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

Update `NVCC_FLAGS` in Makefiles:
- **Volta** (V100): `-arch=sm_70`
- **Turing** (RTX 20 series): `-arch=sm_75`
- **Ampere** (A100, RTX 30 series): `-arch=sm_80`
- **Ada** (RTX 40 series): `-arch=sm_89`

---

## 📊 Program Features

Each program includes:

✅ **Complete, runnable code** - No placeholders
✅ **Error checking** - CUDA_CHECK macro for all API calls
✅ **Performance timing** - cudaEvent-based measurements
✅ **CPU baseline** - CPU version for comparison (where applicable)
✅ **Verification** - Results validation
✅ **Informative output** - Clear performance metrics

### Example Output Format:
```
=== Vector Addition: CPU vs GPU ===

Vector size: 10000000 elements (38.15 MB)

Results:
  CPU Time: 24.53 ms
  GPU Time: 1.87 ms
  Speedup: 13.12x
  Verification: PASSED
```

---

## 🎯 Learning Path

### Beginner Path
1. Start with **Phase 1** programs (hello_world → vector_add → matrix_add)
2. Understand GPU architecture and thread hierarchy
3. Learn memory management basics

### Intermediate Path
4. Move to **Phase 2** (memory optimization)
5. Study **Phase 3** (parallel algorithms)
6. Explore **Phase 4** (advanced memory)

### Advanced Path
7. Master **Phase 5** (complex algorithms)
8. Learn **Phase 6** (concurrency)
9. Optimize with **Phase 7** (performance tuning)

### Expert Path
10. Build **Phase 8** applications
11. Use **Phase 9** modern features

---

## 🐛 Troubleshooting

### "nvcc: command not found"
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### "no CUDA-capable device"
- Verify GPU is installed: `nvidia-smi`
- Check CUDA driver is loaded
- Ensure GPU is not in use by another process

### Compilation errors
- Check compute capability matches your GPU
- Update NVCC_FLAGS in Makefile
- Verify CUDA toolkit version

### Programs run slowly
- Check GPU is not throttling: `nvidia-smi dmon`
- Verify power settings
- Check for background GPU processes

---

## 📈 Performance Expectations

### Typical Speedups (vs CPU)

| Program Type | Expected Speedup |
|--------------|------------------|
| Vector Operations | 10-20x |
| Matrix Operations | 20-50x |
| Reduction | 15-30x |
| N-body | 50-100x |
| Streams (overlap) | 1.5-3x |

*Actual speedup depends on GPU model, problem size, and CPU baseline*

---

## 📚 Additional Resources

### NVIDIA Documentation
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [CUDA Samples](https://github.com/NVIDIA/cuda-samples)

### Tools
- `nvidia-smi` - GPU monitoring
- `nvprof` - Legacy profiler
- `nsys` - Nsight Systems profiler
- `ncu` - Nsight Compute profiler

### Related
- **Colab Notebooks**: `../colab/` - Cloud-based learning
- **FCC Course**: `../collab-fcc-course/` - Structured curriculum

---

## ✨ Summary

- **13 complete CUDA programs** ready to compile and run
- **9 phases** covering basics through advanced topics
- **Production-quality code** with benchmarks
- **Local GPU execution** for maximum performance
- **Easy build system** with Makefiles
- **Comprehensive documentation** in each directory

**Status: READY FOR LOCAL GPU EXECUTION! 🚀**

---

*Generated: 2026-02-24*
*CUDA Toolkit: 11.0+*
*Compute Capability: sm_70+ (Volta and newer)*
