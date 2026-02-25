# Module 14: GPU Programming Preparation

## Overview
Bridge from C++ to GPU programming with CUDA. Understand GPU architecture, memory hierarchies, and write your first GPU programs. This is where everything comes together!

## Topics Covered

### GPU Architecture Fundamentals
- **CPU vs GPU** - Different design philosophies
- **SIMT** (Single Instruction Multiple Thread)
- **GPU cores** - Thousands of simple cores
- **Streaming Multiprocessors (SMs)**
- **Warps** - Groups of 32 threads
- **Thread blocks** and **grids**
- **Compute capability**

### Memory Hierarchies
- **Registers** - Fastest, per-thread
- **Shared memory** - Fast, per-block
- **L1/L2 cache** - Automatic caching
- **Global memory** - Slow, large capacity
- **Constant memory** - Read-only, cached
- **Texture memory** - Specialized read-only
- Memory bandwidth vs latency

### CUDA Basics
- What is CUDA?
- CUDA toolkit installation
- **nvcc** compiler
- CUDA runtime API
- File extensions (.cu, .cuh)

### First CUDA Program
- **__global__** - Kernel functions
- **__device__** - Device functions
- **__host__** - Host functions
- Kernel launch syntax: `kernel<<<grid, block>>>()`
- Thread indexing: `threadIdx`, `blockIdx`, `blockDim`

### Memory Management
- **cudaMalloc** - Allocate on GPU
- **cudaFree** - Free GPU memory
- **cudaMemcpy** - Copy host ↔ device
- **cudaMemcpyHostToDevice**
- **cudaMemcpyDeviceToHost**
- **cudaMemcpyDeviceToDevice**

### Error Handling
- **cudaError_t** - Error type
- **cudaGetLastError** - Check for errors
- **cudaGetErrorString** - Error messages
- Error checking macros
- Synchronous vs asynchronous errors

### Synchronization
- **cudaDeviceSynchronize** - Wait for GPU
- **__syncthreads** - Block-level sync
- Kernel execution is asynchronous

### GPU Selection and Properties
- **cudaGetDeviceCount** - Number of GPUs
- **cudaGetDeviceProperties** - GPU info
- **cudaSetDevice** - Select GPU
- Query compute capability

### When to Use GPU vs CPU
- **GPU best for**:
  - Massive parallelism (millions of operations)
  - Regular memory access patterns
  - High arithmetic intensity
  - Data parallel operations
- **CPU best for**:
  - Sequential tasks
  - Irregular branching
  - Small datasets
  - Low latency requirements

### Performance Considerations
- **Occupancy** - Active warps per SM
- **Memory bandwidth** - Often the bottleneck
- **Coalesced memory access** - Critical!
- **Warp divergence** - Avoid branching
- **Host-device transfer overhead** - Minimize

### CUDA Programming Model
- **Grids, blocks, threads** hierarchy
- Choosing block size (typically 128-512 threads)
- Calculating grid size
- 1D, 2D, 3D thread organization
- Thread indexing patterns

## First CUDA Programs

### 1. Hello World from GPU
```cuda
__global__ void hello() {
    printf("Hello from thread %d\n", threadIdx.x);
}

int main() {
    hello<<<1, 10>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

### 2. Vector Addition
```cuda
__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
```

### 3. Dot Product
```cuda
__global__ void dotProduct(float* a, float* b, float* result, int n) {
    __shared__ float cache[256];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float temp = 0.0f;
    while (tid < n) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = temp;
    __syncthreads();

    // Reduction in shared memory
    // ... (reduction code)
}
```

## Development Environment

### Required Software
- NVIDIA GPU with CUDA support
- CUDA Toolkit (includes nvcc, libraries)
- Compatible C++ compiler
- CMake (for building)

### Compilation
```bash
# Simple CUDA program
nvcc -o program program.cu

# With optimization
nvcc -O3 -arch=sm_75 -o program program.cu

# Using CMake (preferred)
cmake -B build
cmake --build build
```

### Debugging and Profiling
- **cuda-gdb** - GPU debugger
- **compute-sanitizer** - Memory checker
- **Nsight Systems** - System profiling
- **Nsight Compute** - Kernel profiling

## Example Project Structure
```
cuda-project/
├── CMakeLists.txt
├── include/
│   └── kernels.cuh
├── src/
│   ├── main.cpp
│   └── kernels.cu
└── README.md
```

## Common Patterns You'll Use

1. **Memory allocation pattern**:
   - Allocate host memory
   - Allocate device memory
   - Copy host → device
   - Launch kernel
   - Copy device → host
   - Free memory

2. **Error checking pattern**:
   ```cuda
   #define CHECK_CUDA(call) { \
       cudaError_t err = call; \
       if (err != cudaSuccess) { \
           printf("CUDA error: %s\n", cudaGetErrorString(err)); \
           exit(1); \
       } \
   }
   ```

3. **Thread indexing pattern**:
   ```cuda
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx < N) { /* process element idx */ }
   ```

## Why All Previous Modules Matter

| Module | GPU Relevance |
|--------|---------------|
| 1-2: Basics | Understanding C++ syntax for CUDA |
| 3: Pointers | GPU memory management |
| 4-6: OOP | Host-side code organization |
| 7: Templates | Templated kernels, Thrust library |
| 8: STL | Host-side data prep, Thrust |
| 9: Modern C++ | Modern CUDA uses C++14/17 |
| 10: Exceptions | Host-side error handling |
| 11: Threading | Multi-GPU, concurrent streams |
| 12: Build | CMake for CUDA projects |
| 13: Advanced | AoS/SoA, memory patterns |

## Coming Soon

Complete CUDA examples including:
- Setup and installation guide
- Hello World GPU program
- Vector operations
- Matrix multiplication
- Image processing
- Reduction algorithms
- Performance optimization examples

## Estimated Time
20-25 hours

## Prerequisites
**MUST complete Modules 1-13 first!**

All previous concepts come together in GPU programming.

## Next Steps After This Module

1. **CUDA Programming Guide** - Official NVIDIA documentation
2. **GPU Gems books** - Advanced techniques
3. **Thrust library** - High-level GPU programming
4. **cuBLAS/cuFFT** - GPU libraries
5. **Deep learning frameworks** - PyTorch, TensorFlow (if interested)
6. **Parallel algorithms** - Advanced GPU patterns

## Congratulations!

After completing this module, you'll have:
- ✓ Comprehensive C++ knowledge
- ✓ Understanding of GPU architecture
- ✓ Ability to write CUDA programs
- ✓ Foundation for advanced GPU programming

You're ready to become a GPU programmer! 🚀