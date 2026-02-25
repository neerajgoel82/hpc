# CUDA Learning with Google Colab

**56 interactive Jupyter notebooks** for learning CUDA programming without a local GPU.

**100% Free** • **No GPU Required** • **Browser-Based** • **Ready to Run**

---

## Quick Start

```bash
1. Go to https://colab.research.google.com
2. Upload a notebook from notebooks/phase1/
3. Runtime → Change runtime type → GPU (T4)
4. Run all cells and start learning!
```

**First notebook**: `notebooks/phase1/01_hello_world.ipynb`

---

## What You Get

✅ **56 complete notebooks** covering beginner to advanced CUDA
✅ **Working code examples** with full implementations
✅ **Exercise templates** for hands-on practice
✅ **Performance benchmarks** comparing CPU vs GPU
✅ **Real applications** including ray tracing, neural networks, physics
✅ **Modern features** including tensor cores and CUDA graphs
✅ **Free GPU access** via Google Colab (T4, V100, A100)

---

## Google Colab Setup

### Step 1: Access Google Colab
Visit: https://colab.research.google.com

**Requirements**:
- Google account (free)
- Web browser
- Internet connection
- No GPU hardware needed!

### Step 2: Enable GPU Runtime

1. In Colab, click **Runtime** → **Change runtime type**
2. Select **Hardware accelerator**: GPU
3. Choose: **T4 GPU** (free tier) or **V100/A100** (Colab Pro)
4. Click **Save**

### Step 3: Verify GPU Access

Create a new cell and run:
```python
!nvidia-smi
!nvcc --version
```

Expected output:
```
NVIDIA-SMI 525.x.x    Driver Version: 525.x.x    CUDA Version: 12.x
Tesla T4 ...
```

---

## Two Ways to Use CUDA in Colab

### Method 1: %%cu Magic (Recommended)

Install the CUDA extension:
```python
!pip install git+https://github.com/andreinechaev/nvcc4jupyter.git
%load_ext nvcc4jupyter
```

Write CUDA code inline:
```cuda
%%cu
#include <stdio.h>

__global__ void hello() {
    printf("Hello from GPU thread %d!\n", threadIdx.x);
}

int main() {
    hello<<<1, 10>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

### Method 2: Write and Compile .cu Files

```python
# Write CUDA code to file
%%writefile hello.cu
#include <stdio.h>

__global__ void hello() {
    printf("Hello from thread %d!\n", threadIdx.x);
}

int main() {
    hello<<<1, 10>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

```python
# Compile and run
!nvcc hello.cu -o hello
!./hello
```

---

## Directory Structure

```
colab/
├── README.md              # This file
│
└── notebooks/             # 56 Jupyter notebooks
    ├── phase1/           # 6 notebooks - Foundations
    │   ├── 01_hello_world.ipynb
    │   ├── 02_device_query.ipynb
    │   ├── 03_vector_add.ipynb
    │   └── ...
    │
    ├── phase2/           # 6 notebooks - Memory Management
    ├── phase3/           # 6 notebooks - Optimization
    ├── phase4/           # 6 notebooks - Advanced Memory
    ├── phase5/           # 6 notebooks - Advanced Algorithms
    ├── phase6/           # 6 notebooks - Streams & Concurrency
    ├── phase7/           # 5 notebooks - Performance Engineering
    ├── phase8/           # 9 notebooks - Real Applications
    └── phase9/           # 6 notebooks - Modern CUDA
```

---

## Learning Path (16+ Weeks)

### Phase 1: Foundations (Weeks 1-2)
**Goal**: Master basic kernel programming

| Notebook | Topic |
|----------|-------|
| 01 | Hello World - Your first kernel |
| 02 | Device Query - GPU properties |
| 03 | Vector Addition - Parallel operations |
| 04 | Matrix Addition - 2D threading |
| 05 | Thread Indexing - Advanced patterns |

**Skills**: Write kernels, manage GPU memory, understand threading

---

### Phase 2: Memory Management (Weeks 3-4)
**Goal**: Optimize memory access patterns

| Notebook | Topic |
|----------|-------|
| 06 | Memory Basics - cudaMalloc, cudaMemcpy |
| 07 | Bandwidth Benchmarking - PCIe performance |
| 08 | Unified Memory - Managed memory |
| 09 | Shared Memory - On-chip fast memory |
| 10 | Tiled Matrix Multiply - Memory optimization |
| 11 | Memory Coalescing - Access patterns |

**Skills**: Shared memory, coalescing, bandwidth optimization

---

### Phase 3: Optimization (Weeks 5-6)
**Goal**: Write efficient parallel algorithms

| Notebook | Topic |
|----------|-------|
| 12 | Warp Divergence - Branch efficiency |
| 13 | Warp Shuffle - Warp primitives |
| 14 | Occupancy Tuning - Resource optimization |
| 15 | Parallel Reduction - Tree algorithms |
| 16 | Prefix Sum (Scan) - Parallel scan |
| 17 | Histogram - Atomic operations |

**Skills**: Warp operations, reduction, occupancy

---

### Phase 4: Advanced Memory (Weeks 7-8)
**Goal**: Master all memory types

| Notebook | Topic |
|----------|-------|
| 18 | Texture Memory - Cached reads |
| 19 | Constant Memory - Read-only cache |
| 20 | Zero-Copy - Direct host access |
| 21 | Atomics - Thread-safe operations |
| 22 | Cooperative Groups - Flexible sync |
| 23 | Multi-Kernel Sync - Dependencies |

**Skills**: Texture/constant memory, atomics, synchronization

---

### Phase 5: Advanced Algorithms (Weeks 9-10)
**Goal**: Production-quality implementations

| Notebook | Topic |
|----------|-------|
| 24 | Optimized GEMM - Matrix multiply |
| 25 | cuBLAS Integration - BLAS library |
| 26 | Matrix Transpose - Bank conflicts |
| 27 | Bitonic Sort - Sorting networks |
| 28 | Radix Sort - Integer sorting |
| 29 | Thrust Examples - STL for CUDA |

**Skills**: GEMM, cuBLAS, sorting, Thrust

---

### Phase 6: Streams & Concurrency (Week 11)
**Goal**: Asynchronous programming

| Notebook | Topic |
|----------|-------|
| 30 | CUDA Streams - Async execution |
| 31 | Async Pipeline - Overlap compute/transfer |
| 32 | Events and Timing - Precise timing |
| 33 | Multi-GPU Basics - Multiple devices |
| 34 | P2P Transfer - GPU-to-GPU |
| 35 | NCCL Collectives - Multi-GPU communication |

**Skills**: Streams, async operations, multi-GPU

---

### Phase 7: Performance Engineering (Weeks 12-13)
**Goal**: Profile and optimize

| Notebook | Topic |
|----------|-------|
| 36 | Profiling Demo - Nsight tools |
| 37 | Debugging CUDA - Error handling |
| 38 | Kernel Fusion - Reduce overhead |
| 39 | Fast Math - Math intrinsics |
| 40 | Advanced Optimization - Best practices |

**Skills**: Profiling, debugging, optimization

---

### Phase 8: Real Applications (Weeks 14-15)
**Goal**: Build real-world applications

| Notebook | Topic |
|----------|-------|
| 41 | cuFFT - Fast Fourier Transform |
| 42 | cuSPARSE - Sparse matrices |
| 43 | cuRAND - Random numbers |
| 44 | Image Processing - Blur, edge detection |
| 45 | Ray Tracer - Graphics rendering |
| 46 | N-body Simulation - Physics |
| 47 | Neural Network - Deep learning |
| 48 | Molecular Dynamics - Chemistry |
| 49 | Option Pricing - Finance |

**Skills**: Domain libraries, real applications

---

### Phase 9: Modern CUDA (Week 16+)
**Goal**: Latest GPU features

| Notebook | Topic |
|----------|-------|
| 50 | Dynamic Parallelism - Nested kernels |
| 51 | CUDA Graphs - Low-overhead execution |
| 52 | MPS Demo - Multi-Process Service |
| 53 | Mixed Precision - FP16/FP32 |
| 54 | Tensor Cores - WMMA basics |
| 55 | WMMA GEMM - Full tensor core implementation |

**Skills**: Modern features, tensor cores, graphs

---

## Notebook Structure

Each notebook contains:

### 📖 Section 1-2: Introduction
- **Learning objectives** - What you'll learn
- **Concept explanation** - Theory and background

### ✅ Section 3: Example Code (Cell 3)
- **Complete working implementation**
- Real CUDA kernels (not templates!)
- Run to see the concept in action
- Example: N-body with gravitational physics

### 📝 Section 4-5: Practice Exercise (Cell 5)
- **Template for hands-on practice**
- Implement the concept yourself
- Compare with Cell 3 example

### 🎯 Section 6-7: Summary
- **Key takeaways** - Important points
- **Next steps** - Where to go next

### 📓 Section 8: Notes
- **Your personal notes** - Document your learning

**Learning workflow**:
1. Read concept (Cells 1-2)
2. Study example (Cell 3) - Working code ✅
3. Try yourself (Cell 5) - Practice template 📝
4. Compare and learn
5. Take notes (Cell 8)

---

## Google Colab Tips

### Saving Your Work
```python
# Mount Google Drive
from google.collab import drive
drive.mount('/content/drive')

# Save to Drive
!cp notebook.ipynb /content/drive/MyDrive/cuda-learning/
```

### Session Limits
- **Free tier**: 12-hour max session, may disconnect
- **Colab Pro**: Longer sessions, priority GPU access
- **Strategy**: Save work frequently, download outputs

### GPU Availability
- **T4**: Usually available (free tier)
- **V100/A100**: Limited availability (Pro)
- **Peak hours**: May have wait times
- **Tip**: Work during off-peak hours

### Performance
```python
# Check allocated GPU
!nvidia-smi --query-gpu=name,memory.total --format=csv

# Expected on free tier:
# Tesla T4, 15109 MiB
```

---

## Accessing Notebooks

### Method 1: Upload to Colab (Easiest)
1. Download notebook from this repo
2. Go to https://colab.research.google.com
3. File → Upload notebook
4. Enable GPU runtime
5. Run cells

### Method 2: GitHub Integration
1. In Colab: File → Open notebook → GitHub tab
2. Enter: `your-username/repo-name`
3. Browse and open notebooks directly
4. Changes not saved to GitHub (use Drive)

### Method 3: Google Drive Sync
```python
# Clone repo to Drive
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive
!git clone https://github.com/your-username/hpc.git

# Navigate and run
%cd hpc/cuda/samples/colab/notebooks/phase1
```

---

## Common Issues

### Issue: GPU not available
**Solution**: Runtime → Change runtime type → GPU → Save

### Issue: nvcc not found
**Solution**: Colab includes CUDA by default. Restart runtime if needed.

### Issue: Out of memory
**Solution**: Reduce problem size or use gradient checkpointing
```cuda
// Reduce batch size
const int N = 1024 * 1024;  // Instead of 1024 * 1024 * 1024
```

### Issue: Session disconnected
**Solution**: Save work to Drive regularly, use auto-save

### Issue: Compilation errors
**Solution**: Check CUDA syntax, verify compute capability
```python
# Check GPU compute capability
!nvidia-smi --query-gpu=compute_cap --format=csv
```

---

## Advantages of Colab

✅ **Free GPU access** - No hardware investment
✅ **Zero setup** - Works in browser
✅ **Any computer** - Even Chromebooks, tablets
✅ **Latest drivers** - Always up to date
✅ **Easy sharing** - Share notebooks via link
✅ **Collaboration** - Multiple users can edit
✅ **No maintenance** - Google manages infrastructure

---

## Limitations to Know

⚠️ **Session limits** - 12 hours max (free tier)
⚠️ **GPU availability** - Not guaranteed during peak times
⚠️ **No local files** - Must save to Drive
⚠️ **Internet required** - Cannot work offline
⚠️ **Limited storage** - 15 GB Drive free tier
⚠️ **Can't debug** - No cuda-gdb access
⚠️ **Can't profile** - Limited Nsight tool access

**For serious development**: Consider local GPU or cloud compute (AWS, GCP)

---

## Alternative Cloud Options

### Kaggle Notebooks
- Free GPU (30 hours/week)
- Similar to Colab
- Better for competitions
- https://www.kaggle.com/code

### Paperspace Gradient
- Free tier available
- Better GPUs (A100)
- Jupyter notebooks
- https://www.paperspace.com

### Google Cloud / AWS / Azure
- Pay-per-use
- More powerful GPUs
- Full control
- Production workloads

---

## Learning Tips

### For Success
1. **Code along** - Type code yourself, don't just read
2. **Experiment** - Modify examples, break things
3. **Progress daily** - 30 minutes daily beats 3 hours weekly
4. **Complete exercises** - Practice makes perfect
5. **Take notes** - Use Cell 8 in each notebook
6. **Ask questions** - Use forums, Stack Overflow

### Suggested Schedule
- **Weeks 1-2**: Phase 1 (1 hour/day)
- **Weeks 3-4**: Phase 2 (1 hour/day)
- **Weeks 5-6**: Phase 3 (1-2 hours/day)
- **Weeks 7-10**: Phases 4-5 (1-2 hours/day)
- **Weeks 11-13**: Phases 6-7 (1-2 hours/day)
- **Weeks 14-16**: Phases 8-9 (2+ hours/day)

### After Completion
- Build personal projects
- Contribute to open source
- Apply to ML/HPC roles
- Explore advanced topics (cuDNN, TensorRT)

---

## Resources

### Official Documentation
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [cuBLAS](https://docs.nvidia.com/cuda/cublas/)
- [cuFFT](https://docs.nvidia.com/cuda/cufft/)

### Community
- [NVIDIA Forums](https://forums.developer.nvidia.com/)
- [Stack Overflow - CUDA](https://stackoverflow.com/questions/tagged/cuda)
- [Reddit r/CUDA](https://www.reddit.com/r/CUDA/)

### Books
- "CUDA by Example" by Sanders & Kandrot
- "Programming Massively Parallel Processors" by Hwu et al.
- "CUDA Handbook" by Wilt

---

## Getting Help

If you encounter issues:
1. **Check error messages** - CUDA errors are descriptive
2. **Search Stack Overflow** - Likely answered already
3. **Read documentation** - CUDA Programming Guide
4. **Ask in forums** - NVIDIA Developer Forums
5. **Check notebook comments** - Explanations included

---

## Statistics

- **Total notebooks**: 56
- **Working examples**: 56 (all have real code in Cell 3)
- **Practice exercises**: 56 (all have templates in Cell 5)
- **Lines of code**: ~15,000+
- **Concepts covered**: 80+
- **Learning time**: 16+ weeks
- **Cost**: $0 (with Colab free tier)

---

## Next Steps

1. **Verify GPU access**: Run `!nvidia-smi` in Colab
2. **Open first notebook**: Upload `notebooks/phase1/01_hello_world.ipynb`
3. **Enable GPU runtime**: Runtime → Change runtime type → GPU
4. **Run all cells**: Runtime → Run all
5. **Continue to next**: Follow the phases sequentially

---

**Ready to start?** Open `notebooks/phase1/01_hello_world.ipynb` in Google Colab! 🚀

**No GPU? No problem!** Start learning CUDA today, completely free! 💯
