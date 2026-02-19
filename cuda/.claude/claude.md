# CUDA Programming - Claude Instructions

## Repository Structure

CUDA samples are organized into:
- **colab/**: Google Colab notebooks for cloud-based learning
- **collab-fcc-course/**: FreeCodeCamp CUDA course materials
- **local/**: Local CUDA samples for native GPU execution
- **notebooks/**: Additional Jupyter notebooks for CUDA learning

## Compilation Standards

### NVCC Compilation
```bash
nvcc -arch=sm_XX file.cu -o output
```

- Use appropriate compute capability for target GPU
- `-arch=sm_70` for Volta, `-arch=sm_80` for Ampere, etc.
- Add `-Xcompiler -Wall` for host code warnings

### Common Flags
- `-O2` or `-O3` for optimized code
- `--ptxas-options=-v` to see register/memory usage
- `-lineinfo` for debugging and profiling

## Coding Style

### File Structure
```cpp
// Order:
1. Includes
2. Kernel definitions (__global__)
3. Device functions (__device__) if needed
4. Host functions and main()
```

### Kernel Naming
- Descriptive kernel names: `vectorAddKernel`, `matrixMulKernel`
- Device functions: prefix with `device_` if helpful
- Use `__global__` for kernels, `__device__` for device-only functions

### Code Organization
```cpp
__global__ void myKernel(float* d_out, const float* d_in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // kernel logic
    }
}

int main() {
    // 1. Allocate host memory
    // 2. Allocate device memory (cudaMalloc)
    // 3. Copy host to device (cudaMemcpy H2D)
    // 4. Launch kernel
    // 5. Copy device to host (cudaMemcpy D2H)
    // 6. Free memory
    // 7. Check for errors
}
```

## Common Patterns

### Thread Indexing
```cpp
// 1D
int idx = blockIdx.x * blockDim.x + threadIdx.x;

// 2D
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int idx = y * width + x;

// Always check bounds
if (idx < n) { /* work */ }
```

### Memory Management
```cpp
// Allocation
float* d_data;
cudaMalloc(&d_data, size * sizeof(float));

// Copy to device
cudaMemcpy(d_data, h_data, size * sizeof(float), cudaMemcpyHostToDevice);

// Copy to host
cudaMemcpy(h_data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost);

// Free
cudaFree(d_data);
```

### Error Checking
```cpp
// For kernel launches
kernel<<<blocks, threads>>>(args);
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("Kernel launch error: %s\n", cudaGetErrorString(err));
}

// For CUDA API calls
cudaError_t err = cudaMalloc(&d_ptr, size);
if (err != cudaSuccess) {
    printf("cudaMalloc error: %s\n", cudaGetErrorString(err));
}

// Or use error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)
```

### Kernel Launch Configuration
```cpp
// Calculate grid/block dimensions
int threads = 256;  // or 128, 512 (multiples of 32)
int blocks = (n + threads - 1) / threads;  // ceiling division

// Launch
myKernel<<<blocks, threads>>>(args);
```

## Complexity Levels

### Beginner Samples
- Simple vector operations (add, multiply)
- Basic memory transfers
- 1D thread indexing
- Error checking can be basic

### Intermediate Samples
- Matrix operations
- 2D thread indexing
- Shared memory usage
- Thread synchronization (`__syncthreads()`)
- Proper error handling

### Advanced Samples
- Optimized memory access patterns (coalescing)
- Warp-level operations
- Atomic operations
- Multiple streams
- Unified memory
- Performance profiling

## What NOT to Do

- Don't ignore bounds checking in kernels
- Don't forget to check CUDA errors (both API and kernels)
- Don't hardcode grid/block dimensions without explanation
- Don't ignore memory coalescing in performance samples
- Don't use `cudaDeviceSynchronize()` excessively
- Don't forget to free device memory

## Colab vs Local

### Colab Notebooks
- Include `!nvcc --version` to check CUDA availability
- Use cell magic for compilation: `%%writefile file.cu`
- Good for learning without local GPU
- May have limited compute capability

### Local Samples
- Assume CUDA toolkit installed
- Can use Makefiles
- Performance testing requires local GPU
- More suitable for advanced optimization work

## Performance Considerations

### Memory Access
- Coalesced access patterns (stride-1 access)
- Minimize global memory accesses
- Use shared memory for frequently accessed data
- Consider memory bank conflicts in shared memory

### Occupancy
- Balance threads per block with register usage
- Check occupancy with `--ptxas-options=-v`
- More threads isn't always better

### Optimization Checklist
1. Profile first (nvprof, Nsight)
2. Optimize memory access patterns
3. Use appropriate memory types (shared, constant)
4. Minimize divergence
5. Consider warp-level operations

## When Adding New Samples
1. Decide if it's colab or local
2. Start with working CPU version for comparison
3. Add comprehensive error checking
4. Document performance characteristics
5. Include comments on optimization strategies

## Testing
- Verify on actual GPU hardware when possible
- Compare results with CPU version
- Check for CUDA errors
- Profile with nvprof or Nsight
- Test with different input sizes

## Documentation
- Explain parallelization strategy
- Document thread/block organization
- Note memory usage patterns
- Include performance measurements when relevant
- Comment on optimization opportunities
