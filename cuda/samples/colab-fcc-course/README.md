# FreeCodeCamp CUDA Course - Google Colab Edition

**Learn GPU programming without a local GPU!**

This directory contains **25 Jupyter notebooks** adapted for Google Colab from the [FreeCodeCamp CUDA Programming Course](https://www.youtube.com/watch?v=86FAWCzIe_4). Run professional CUDA code in your browser with free GPU access.

**100% Free** • **No GPU Hardware Required** • **Browser-Based** • **Ready to Run**

---

## Quick Start

```bash
1. Go to https://colab.research.google.com
2. Upload a notebook from module5/
3. Runtime → Change runtime type → T4 GPU
4. Run all cells and start learning!
```

**First notebook**: `module5/01_CUDA_Basics_01_idxing.ipynb`

---

## What You'll Learn

✅ **CUDA fundamentals** - Thread hierarchy, memory management, kernel programming
✅ **Optimization techniques** - Shared memory, tiling, profiling, atomic operations
✅ **CUDA libraries** - cuBLAS, cuDNN, cuBLASLt for production performance
✅ **Advanced optimization** - Loop unrolling, register blocking, instruction-level parallelism
✅ **Triton** - High-level GPU programming (10x less code than CUDA)
✅ **PyTorch extensions** - Integrate custom CUDA kernels with PyTorch
✅ **Production skills** - Profiling, debugging, benchmarking

---

## Course Modules

### Module 5: Writing Your First Kernels (10 notebooks)
**Duration**: 8-12 hours | **Difficulty**: Beginner to Intermediate

Essential CUDA fundamentals covering:
- Thread indexing and calculation
- Vector and matrix operations
- Profiling with NVTX
- Naive vs tiled matrix multiplication
- Atomic operations
- CUDA streams and asynchronous execution

**Start here**: `module5/01_CUDA_Basics_01_idxing.ipynb`
**Key notebook**: `03_Profiling_02_tiled_matmul.ipynb` (10-20x speedup!)

[📖 Module 5 README](module5/README.md)

---

### Module 6: CUDA APIs (cuBLAS, cuDNN) (9 notebooks)
**Duration**: 8-10 hours | **Difficulty**: Intermediate

Leverage NVIDIA's highly optimized libraries:
- **cuBLAS**: FP16 and FP32 matrix operations
- **cuBLASLt**: Flexible GEMM with Tensor Cores
- **cuBLASXt**: Multi-GPU linear algebra
- **cuDNN**: Deep learning operations (conv, activations)
- **CUTLASS**: Customizable GEMM templates

**Start here**: `module6/01_CUBLAS_01_cuBLAS_01_Hgemm_Sgemm.ipynb`
**Performance**: Achieve 100-300 TFLOPS with Tensor Cores

[📖 Module 6 README](module6/README.md)

---

### Module 7: Optimizing Matrix Multiplication (1 notebook)
**Duration**: 2-3 hours | **Difficulty**: Advanced

Advanced optimization techniques:
- Loop unrolling with `#pragma unroll`
- Register blocking (4x4, 8x8 tiles per thread)
- Vectorized memory access (float4)
- Instruction-level parallelism (ILP)

**Notebook**: `module7/unrolling_example.ipynb`
**Goal**: Achieve 60-80% of cuBLAS FP32 performance

[📖 Module 7 README](module7/README.md)

---

### Module 8: Triton - High-Level GPU Programming (3 notebooks)
**Duration**: 4-6 hours | **Difficulty**: Intermediate

Modern GPU programming with Python:
- Triton programming model (block-based)
- Vector addition with benchmarking
- Softmax implementation (CUDA vs Triton comparison)
- Automatic optimization and autotuning

**Start here**: `module8/01_triton_vector_add.ipynb`
**Benefit**: 80-95% of CUDA performance with 10x less code

[📖 Module 8 README](module8/README.md)

---

### Module 9: PyTorch CUDA Extensions (2 notebooks)
**Duration**: 3-5 hours | **Difficulty**: Intermediate to Advanced

Integrate custom CUDA with PyTorch:
- Create PyTorch C++ extensions
- Implement forward and backward passes
- Support automatic differentiation (autograd)
- Production deployment patterns

**Notebooks**:
- `polynomial_cuda.ipynb` - Basic extension
- `pytorch_extension_demo.ipynb` - Complete guide

[📖 Module 9 README](module9/README.md)

---

## Learning Paths

### 🎯 Beginner Path (Recommended)
```
Module 5 (Notebooks 1-4) → Module 5 (Notebook 7: Tiling)
        ↓
Module 8 (Triton intro) → Module 9 (PyTorch extensions)
```
**Focus**: Core concepts and modern tools

### 🚀 Performance Path
```
Module 5 (Profiling) → Module 7 (Optimization) → Module 6 (Libraries)
        ↓
Module 5 (Streams & Atomics)
```
**Focus**: Maximum performance and optimization

### 🔬 Research Path
```
Module 5 (Fundamentals) → Module 6 (Understanding libraries)
        ↓
Module 8 (Triton prototyping) → Module 9 (PyTorch integration)
```
**Focus**: Novel implementations and rapid iteration

---

## Directory Structure

```
collab-fcc-course/
├── README.md           # This file
│
├── module5/           # 10 notebooks - Writing Your First Kernels
│   ├── README.md
│   ├── 01_CUDA_Basics_01_idxing.ipynb
│   ├── 02_Kernels_00_vector_add_v1.ipynb
│   ├── 02_Kernels_02_matmul.ipynb
│   ├── 03_Profiling_02_tiled_matmul.ipynb
│   ├── 04_Atomics_00_atomicAdd.ipynb
│   └── 05_Streams_02_stream_advanced.ipynb
│
├── module6/           # 9 notebooks - CUDA APIs
│   ├── README.md
│   ├── 01_CUBLAS_01_cuBLAS_01_Hgemm_Sgemm.ipynb
│   ├── 01_CUBLAS_02_cuBLASLt_01_LtMatmul.ipynb
│   ├── 02_CUDNN_01_Conv2d_NCHW.ipynb
│   └── optional_CUTLASS_compare.ipynb
│
├── module7/           # 1 notebook - Advanced Optimization
│   ├── README.md
│   └── unrolling_example.ipynb
│
├── module8/           # 3 notebooks - Triton
│   ├── README.md
│   ├── 01_triton_vector_add.ipynb
│   ├── 02_softmax.ipynb
│   └── 03_triton_softmax.ipynb
│
└── module9/           # 2 notebooks - PyTorch Extensions
    ├── README.md
    ├── polynomial_cuda.ipynb
    └── pytorch_extension_demo.ipynb
```

---

## Google Colab Setup

### Step 1: Access Google Colab
Visit: https://colab.research.google.com

**Requirements**:
- Google account (free tier works!)
- Web browser
- Internet connection
- **No GPU hardware needed!**

### Step 2: Enable GPU Runtime

1. In Colab, click **Runtime** → **Change runtime type**
2. Select **Hardware accelerator**: GPU
3. Choose: **T4 GPU** (free tier) or **V100/A100** (Colab Pro)
4. Click **Save**

### Step 3: Verify GPU Access

Run in a cell:
```python
!nvidia-smi
!nvcc --version
```

Expected output:
```
Tesla T4, 16GB
CUDA Version: 12.x
```

### Step 4: Upload and Run Notebook

1. Click **File** → **Upload notebook**
2. Select a notebook from module5/
3. Run all cells: **Runtime** → **Run all**

---

## Course Statistics

- **Total notebooks**: 25
- **Modules**: 5 (Module 5-9)
- **Lines of CUDA code**: ~5,000+
- **Concepts covered**: 50+
- **Learning time**: 25-40 hours
- **Cost**: $0 (with Colab free tier)

---

## Prerequisites

### Knowledge
- Python programming
- Basic C/C++ syntax
- Understanding of arrays and matrices
- Basic parallel computing concepts (helpful but not required)

### Tools
- Google account
- Web browser
- Internet connection

---

## Time Estimates

| Module | Content | Fast Pace | Moderate | Relaxed |
|--------|---------|-----------|----------|---------|
| **Module 5** | CUDA Basics | 1 week | 1.5 weeks | 2 weeks |
| **Module 6** | Libraries | 1 week | 1.5 weeks | 2 weeks |
| **Module 7** | Optimization | 2-3 hours | 4-6 hours | 1 week |
| **Module 8** | Triton | 4-6 hours | 6-8 hours | 1.5 weeks |
| **Module 9** | PyTorch | 3-4 hours | 5-6 hours | 1 week |
| **Total** | | 2.5 weeks | 4-5 weeks | 7-8 weeks |

---

## Google Colab Tips

### Free Tier Limits
- **12-hour session limit**: Save work frequently
- **90-minute idle disconnect**: Keep browser tab active
- **GPU availability**: May wait during peak hours
- **Shared resources**: Performance varies

### Save Your Work
```python
# Download modified notebook
from google.colab import files
files.download('notebook.ipynb')
```

### Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')

# Save to Drive
!cp notebook.ipynb /content/drive/MyDrive/cuda-learning/
```

### Check GPU Info
```python
!nvidia-smi
# Shows: GPU type, memory, utilization
```

### Faster Recompilation
```python
# Use ccache for faster builds
!export CCACHE_DIR=/tmp/ccache
```

---

## Common Issues

### GPU Not Available
**Solution**: Runtime → Change runtime type → GPU → Save

### Runtime Disconnected
**Solution**: Colab idle timeout. Reconnect and re-run cells.

### CUDA Out of Memory
**Solution**: Reduce tensor/matrix sizes in examples. Restart runtime.

### nvcc Not Found
**Solution**: Run setup cells in notebook. nvcc4jupyter handles compilation.

### Slow Compilation
**Solution**: First compilation is slow. Subsequent runs are faster (cached).

### Import Errors
**Solution**: Re-run installation cells. Verify GPU is enabled.

---

## Attribution & Resources

### Original Course
- **YouTube**: [CUDA Programming on FreeCodeCamp](https://www.youtube.com/watch?v=86FAWCzIe_4)
- **Repository**: [Infatoshi/cuda-course](https://github.com/Infatoshi/cuda-course)
- **Duration**: ~10 hours video content

### Official Documentation
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [cuDNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/)
- [Triton Documentation](https://triton-lang.org/)
- [PyTorch Extensions Guide](https://pytorch.org/tutorials/advanced/cpp_extension.html)

### Community
- [CUDA MODE Discord](https://discord.gg/cudamode) - Active GPU programming community
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)
- Check original course for Discord/community links

---

## Success Tips

1. **Run Every Cell**: Don't just read - execute and observe
2. **Modify Code**: Change parameters, break things, fix them
3. **Do Exercises**: Each notebook has practice problems
4. **Take Notes**: Use the Notes section in each notebook
5. **Watch Video**: Follow [YouTube course](https://www.youtube.com/watch?v=86FAWCzIe_4) for details
6. **Ask Questions**: Use course community when stuck
7. **Profile Code**: Always measure before optimizing
8. **Compare Implementations**: See how different approaches perform

---

## What's Next?

After completing this course:

### 🏗️ Build Projects
- Neural network from scratch in CUDA
- Image/video processing pipeline
- Physics simulations (N-body, molecular dynamics)
- Custom PyTorch operators for research

### 📚 Explore Advanced Topics
- Multi-GPU programming with NCCL
- Tensor Cores for AI (100+ TFLOPS)
- Dynamic parallelism
- CUDA Graphs for low-overhead execution
- Mixed precision training

### 📖 Read Research Papers
- Flash Attention (memory-efficient attention)
- Efficient transformer implementations
- Sparse matrix operations
- Custom CUDA kernels for novel architectures

### 🤝 Contribute to Open Source
- PyTorch
- Triton
- RAPIDS
- GPU-accelerated libraries

---

## FAQ

**Q: Do I need a local GPU?**
A: No! Run entirely on Google Colab's free GPUs.

**Q: What GPU does Colab provide?**
A: Usually T4 (16GB), sometimes V100 or P100. Colab Pro gets A100.

**Q: Can I use Kaggle or Paperspace?**
A: Yes! Similar setup - just enable GPU in their settings.

**Q: I'm new to parallel programming. Can I learn?**
A: Absolutely! Start with Module 5, Notebook 1. Take it slow.

**Q: How long does the course take?**
A: Your pace. Could be 2 weeks intensive or 2 months relaxed.

**Q: Do I need C++ experience?**
A: Basic C helps. Notebooks explain CUDA-specific concepts.

**Q: Will this teach me deep learning?**
A: This teaches GPU programming. You can apply it to deep learning.

**Q: Is this better than the YouTube course alone?**
A: Complement the video with hands-on practice in these notebooks.

---

## Next Steps

**Ready to start?** 

1. Open `module5/01_CUDA_Basics_01_idxing.ipynb` in Google Colab
2. Enable GPU runtime
3. Run all cells
4. Begin your GPU programming journey!

---

**Remember**: The best way to learn GPU programming is by doing. Run the code, modify it, break it, fix it. That's how you master CUDA.

**Happy GPU Programming!** 🚀⚡

For detailed module information, see individual module READMEs.
