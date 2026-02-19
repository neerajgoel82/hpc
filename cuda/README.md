# CUDA Programming

Learn GPU programming with NVIDIA CUDA for high-performance parallel computing.

## Structure

```
cuda/
├── samples/
│   ├── colab/              # Google Colab notebooks
│   ├── collab-fcc-course/  # FreeCodeCamp CUDA course
│   ├── local/              # Local GPU samples
│   └── convert_cuda_to_colab.py  # Conversion utility
├── projects/               # Complete CUDA applications
└── notebooks/              # CUDA learning notebooks
```

## Overview

CUDA samples are organized by environment:

### Colab
- Cloud-based learning without local GPU
- Jupyter notebooks with embedded CUDA code
- Good for beginners and experimentation
- Limited compute capability

### Local
- Native GPU execution
- Better performance
- Full CUDA toolkit features
- Requires NVIDIA GPU

### FCC Course
- Materials from FreeCodeCamp CUDA course
- Structured learning path
- Hands-on examples

## Getting Started

### Prerequisites

#### For Local Development
```bash
# Check for NVIDIA GPU
nvidia-smi

# Install CUDA Toolkit
# Visit: https://developer.nvidia.com/cuda-downloads
# Or on Ubuntu/Debian:
sudo apt-get install nvidia-cuda-toolkit

# Verify installation
nvcc --version
```

#### For Colab
- Google account
- No local GPU required
- Open notebooks directly in Google Colab

### Compiling CUDA Code

```bash
# Basic compilation
nvcc file.cu -o output

# With compute capability (check your GPU)
nvcc -arch=sm_70 file.cu -o output  # Volta
nvcc -arch=sm_80 file.cu -o output  # Ampere

# With warnings and optimization
nvcc -arch=sm_70 -Xcompiler -Wall -O2 file.cu -o output

# Check resource usage
nvcc --ptxas-options=-v file.cu -o output
```

### Finding Your GPU's Compute Capability
```bash
# Run nvidia-smi to see GPU model
nvidia-smi

# Or use deviceQuery sample from CUDA SDK
./deviceQuery
```

Common compute capabilities:
- **sm_70**: Volta (V100)
- **sm_75**: Turing (RTX 20 series)
- **sm_80**: Ampere (A100, RTX 30 series)
- **sm_86**: Ampere (RTX 3090)
- **sm_89**: Ada Lovelace (RTX 40 series)

## Learning Path

### 1. Fundamentals
- Understand GPU architecture
- Learn kernel syntax
- Master thread indexing
- Practice memory transfers

### 2. Memory Management
- Global memory
- Shared memory
- Constant memory
- Memory coalescing

### 3. Optimization
- Minimize memory transfers
- Coalesced memory access
- Shared memory usage
- Occupancy optimization

### 4. Advanced Topics
- Streams and concurrency
- Unified memory
- Warp-level operations
- Atomic operations

## Code Structure

### Typical CUDA Program
```cpp
#include <cuda_runtime.h>
#include <stdio.h>

// Kernel definition
__global__ void myKernel(float* d_out, const float* d_in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_out[idx] = d_in[idx] * 2.0f;
    }
}

int main() {
    // 1. Setup
    int n = 1024;
    size_t bytes = n * sizeof(float);

    // 2. Allocate host memory
    float *h_in, *h_out;
    h_in = (float*)malloc(bytes);
    h_out = (float*)malloc(bytes);

    // 3. Initialize data
    for (int i = 0; i < n; i++) {
        h_in[i] = (float)i;
    }

    // 4. Allocate device memory
    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    // 5. Copy to device
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    // 6. Launch kernel
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    myKernel<<<blocks, threads>>>(d_out, d_in, n);

    // 7. Copy to host
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    // 8. Cleanup
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);

    return 0;
}
```

## Error Checking

Always check CUDA errors:
```cpp
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Usage
CUDA_CHECK(cudaMalloc(&d_ptr, size));
```

## Performance Tips

### Memory Access
- Coalesced access (stride-1)
- Minimize global memory accesses
- Use shared memory for reused data
- Avoid bank conflicts

### Kernel Configuration
- Threads per block: multiples of 32 (warp size)
- Common choices: 128, 256, 512
- Balance occupancy vs resource usage

### Optimization Workflow
1. Write correct code first
2. Profile with nvprof or Nsight
3. Identify bottlenecks
4. Optimize memory access patterns
5. Consider shared memory
6. Profile again

## Profiling Tools

```bash
# Legacy profiler
nvprof ./program

# Newer tools
nsys profile ./program
ncu ./program
```

## Resources

### Documentation
- **[samples/START_HERE.md](samples/START_HERE.md)** - Quick start guide
- **[samples/CURRICULUM_COMPLETE.md](samples/CURRICULUM_COMPLETE.md)** - Complete CUDA curriculum
- **[samples/QUICK_START.md](samples/QUICK_START.md)** - Setup instructions
- **[samples/FCC_CUDA_COLAB_COMPLETE.md](samples/FCC_CUDA_COLAB_COMPLETE.md)** - FreeCodeCamp course guide
- **samples/README.md** - Overview of sample organization

### Learning Materials
- **samples/colab/** - 81+ Jupyter notebooks for Google Colab
  - Learn CUDA without local GPU
  - Complete curriculum in interactive notebooks
  - GPU provided by Google Colab
- **samples/collab-fcc-course/** - FreeCodeCamp CUDA course
  - Modules 5-9 covering advanced topics
  - Profiling, streams, atomics, and more
- **samples/local/** - Native GPU samples
  - For systems with NVIDIA GPUs
  - Projects and examples for local execution

### Utilities
- **samples/convert_cuda_to_colab.py** - Convert .cu files to Colab notebooks

## Common Pitfalls

- Forgetting bounds checking in kernels
- Not checking CUDA errors
- Inefficient memory access patterns
- Too many/too few threads per block
- Not freeing device memory

## Sample Count

### Colab Notebooks
- **81+ Jupyter notebooks** across multiple topics
- Interactive learning without local GPU
- Complete curriculum coverage

### FCC Course
- **Modules 5-9** with advanced CUDA topics
- Profiling, streams, atomics, unified memory
- Production-ready patterns

### Local Samples
- Native GPU examples
- Projects for local execution
- Performance benchmarking

**Total**: 105+ files (notebooks, documentation, scripts)

## Next Steps

1. **For Beginners (No GPU)**:
   - Start with [samples/colab/](samples/colab/) notebooks
   - Open in Google Colab (free GPU access)
   - Follow the curriculum from START_HERE.md

2. **For Local GPU**:
   - Explore [samples/local/](samples/local/) examples
   - Compile and run on your NVIDIA GPU
   - Benchmark and profile your code

3. **Advanced Learning**:
   - Work through FCC course modules
   - Study optimization techniques
   - Build complete applications in [projects/](projects/)

4. **Cross-Language Comparison**:
   - Compare with [C++](../cpp/) implementations
   - Understand when to use GPU vs CPU
   - Measure speedup and efficiency
