# Getting Started with FCC CUDA Course on Google Colab

**Quick start guide for learning CUDA programming without local GPUs!**

---

## What You'll Learn

This collection of **26 Jupyter notebooks** covers the complete FreeCodeCamp CUDA Programming Course, adapted for Google Colab:

- ✅ CUDA fundamentals and thread hierarchy
- ✅ Memory management and optimization
- ✅ High-performance kernels (matrix multiplication, reductions)
- ✅ CUDA libraries (cuBLAS, cuDNN)
- ✅ Profiling and debugging
- ✅ Triton for high-level GPU programming
- ✅ PyTorch CUDA extensions

---

## Setup (5 Minutes)

### Step 1: Open Google Colab
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Sign in with your Google account (free account works!)

### Step 2: Upload Your First Notebook
1. Click **File** → **Upload notebook**
2. Navigate to `collab-fcc-course/module5/`
3. Upload `01_CUDA_Basics_01_idxing.ipynb`

### Step 3: Enable GPU
1. In Colab, click **Runtime** → **Change runtime type**
2. Under **Hardware accelerator**, select **T4 GPU**
3. Click **Save**

### Step 4: Run the Notebook
1. Click **Runtime** → **Run all**
2. Wait for the setup cells to complete
3. See your first CUDA program run on a real GPU! 🎉

---

## Course Structure

### 📁 Module 5: Writing Your First Kernels (10 notebooks)
**Essential CUDA fundamentals**

- Thread indexing and hierarchy
- Vector addition (CPU vs GPU)
- Matrix multiplication
- Profiling with NVTX
- Shared memory tiling
- Atomic operations
- CUDA streams

**Start here:** `module5/01_CUDA_Basics_01_idxing.ipynb`

### 📁 Module 6: CUDA APIs (9 notebooks)
**Leverage optimized NVIDIA libraries**

- cuBLAS for linear algebra (Hgemm, Sgemm)
- cuBLASLt for advanced matrix operations
- cuBLASXt for multi-GPU
- cuDNN for deep learning (activations, convolutions)
- CUTLASS for custom GEMM kernels

**Start here:** `module6/01_CUBLAS_01_cuBLAS_01_Hgemm_Sgemm.ipynb`

### 📁 Module 7: Optimizing Matrix Multiplication (1 notebook)
**Deep dive into performance optimization**

- Loop unrolling with `#pragma unroll`
- Register usage optimization
- Occupancy tuning
- PTX assembly analysis

**Notebook:** `module7/unrolling_example.ipynb`

### 📁 Module 8: Triton (3 notebooks)
**High-level GPU programming with Python**

- Triton programming model
- Vector addition with benchmarking
- Softmax implementation
- Comparison with CUDA

**Start here:** `module8/01_triton_vector_add.ipynb`

### 📁 Module 9: PyTorch CUDA Extensions (2 notebooks)
**Integrate custom CUDA code with PyTorch**

- Custom activation functions
- PyBind11 integration
- JIT compilation
- Performance comparison

**Notebook:** `module9/pytorch_extension_demo.ipynb`

---

## Learning Path

### 🎯 Beginner Path (Start Here)
1. Module 5, Notebooks 1-4: CUDA basics
2. Module 5, Notebook 7: Tiled matrix multiplication
3. Module 8, Notebook 1: Triton introduction
4. Module 9: PyTorch extensions

### 🚀 Performance Path (Optimization Focus)
1. Module 5, Notebooks 5-7: Profiling and optimization
2. Module 7: Advanced matmul optimization
3. Module 6: CUDA libraries
4. Module 5, Notebooks 8-10: Advanced features

### 🔬 Research Path (Novel Implementations)
1. Module 5: CUDA fundamentals
2. Module 6: Understanding optimized libraries
3. Module 8: Triton for rapid prototyping
4. Module 9: PyTorch integration

---

## Tips for Success

### 1. Run Every Cell
Don't just read - execute the code and observe outputs.

### 2. Modify and Experiment
Change parameters, sizes, block configurations. Break things and fix them!

### 3. Do the Exercises
Each notebook has exercises. They reinforce learning.

### 4. Take Notes
Use the Notes section at the end of each notebook.

### 5. Watch the Video
Follow along with the [YouTube course](https://www.youtube.com/watch?v=86FAWCzIe_4) for detailed explanations.

### 6. Use the Discord/Community
The original course has a community - use it for questions!

---

## Google Colab Tips

### Free Tier Limits
- **12-hour session limit**: Save your work frequently
- **GPU availability**: May need to wait during peak times
- **90-minute idle disconnect**: Keep the browser tab active

### Saving Your Work
```python
# Download modified notebooks
from google.colab import files
files.download('notebook.ipynb')
```

### Faster Compilation
```python
# Use Colab's ccache for faster recompilation
!export CCACHE_DIR=/tmp/ccache
```

### Checking GPU Info
```python
!nvidia-smi
# Shows: GPU type (T4), memory, utilization
```

---

## Troubleshooting

### "Runtime disconnected"
- Colab idle timeout: Reconnect and re-run cells
- Save progress regularly!

### "CUDA out of memory"
- Reduce tensor sizes in examples
- Restart runtime: Runtime → Restart runtime

### "nvcc not found"
- The nvcc4jupyter plugin handles compilation
- Make sure you ran the setup cells

### Slow compilation
- First compilation is always slow (cold cache)
- Subsequent runs are faster

### Import errors
- Re-run the installation cells
- Check that GPU is enabled

---

## Course Resources

### Original Course
- **YouTube**: [CUDA Programming on FreeCodeCamp](https://www.youtube.com/watch?v=86FAWCzIe_4)
- **Repository**: [Infatoshi/cuda-course](https://github.com/Infatoshi/cuda-course)

### CUDA Documentation
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [cuDNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/)

### Triton
- [Triton Language](https://triton-lang.org/)
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/)

### PyTorch Extensions
- [PyTorch C++/CUDA Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html)
- [PyTorch C++ API](https://pytorch.org/cppdocs/)

### Community
- Check the original course for Discord/community links
- [CUDA MODE Discord](https://discord.gg/cudamode) - Active GPU programming community

---

## What's Next?

After completing these notebooks:

1. **Build Projects**: Apply what you learned
   - Neural network from scratch in CUDA
   - Image processing pipeline
   - Physics simulations
   - Custom PyTorch operators

2. **Explore Advanced Topics**
   - Multi-GPU programming (NCCL)
   - Tensor Cores for AI
   - Dynamic parallelism
   - CUDA Graphs

3. **Read Research Papers**
   - Flash Attention
   - Efficient attention mechanisms
   - Sparse matrix operations

4. **Contribute to Open Source**
   - PyTorch
   - Triton
   - GPU libraries

---

## FAQ

**Q: Do I need a local GPU?**
A: No! These notebooks run entirely on Google Colab's free GPUs.

**Q: What GPU does Colab provide?**
A: Usually T4 (16GB VRAM), sometimes V100 or P100.

**Q: Can I use Colab Pro?**
A: Yes! Pro gives longer sessions and better GPUs (A100, V100).

**Q: Will this work on Kaggle/Paperspace?**
A: Yes! Similar setup, just enable GPU in their settings.

**Q: I'm new to parallel programming. Can I still learn?**
A: Absolutely! Start with Module 5, Notebook 1. Take it slow.

**Q: How long does the course take?**
A: At your own pace. Could be 2 weeks intensive or 2 months relaxed.

**Q: Do I need C++ experience?**
A: Basic C knowledge helps. The notebooks explain CUDA-specific concepts.

---

## Support

If you find issues with the notebooks:
1. Check the troubleshooting section above
2. Review the original course materials
3. Ask in the course community

---

## Let's Go! 🚀

Ready to start? Open `module5/01_CUDA_Basics_01_idxing.ipynb` in Google Colab and begin your GPU programming journey!

**Remember**: The best way to learn is by doing. Run the code, modify it, break it, fix it. That's how you master CUDA programming.

Happy GPU programming! ⚡
