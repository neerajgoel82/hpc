#!/usr/bin/env python3
"""
Fix the remaining 6 .cu files with better implementations.
"""

FILES = {
    'local/phase4/20_zero_copy.cu': '''#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \\
    do { \\
        cudaError_t err = call; \\
        if (err != cudaSuccess) { \\
            fprintf(stderr, "CUDA error at %s:%d: %s\\n", \\
                    __FILE__, __LINE__, cudaGetErrorString(err)); \\
            exit(EXIT_FAILURE); \\
        } \\
    } while(0)

__global__ void processKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = sqrtf(data[idx]) * 2.0f + 1.0f;
    }
}

int main() {
    printf("=== Zero-Copy Memory ===\\n\\n");

    int n = 1 << 20;
    size_t size = n * sizeof(float);

    // Check if device supports mapped memory
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    if (!prop.canMapHostMemory) {
        printf("Device doesn't support mapped memory\\n");
        return 1;
    }

    // Allocate zero-copy (mapped) memory
    float *h_data;
    CUDA_CHECK(cudaHostAlloc(&h_data, size, cudaHostAllocMapped));

    for (int i = 0; i < n; i++) h_data[i] = i + 1.0f;

    // Get device pointer to mapped memory
    float *d_data;
    CUDA_CHECK(cudaHostGetDevicePointer(&d_data, h_data, 0));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    processKernel<<<blocks, threads>>>(d_data, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // Verify - host can directly access results
    bool correct = true;
    for (int i = 0; i < 1000; i++) {
        float expected = sqrtf(i + 1.0f) * 2.0f + 1.0f;
        if (abs(h_data[i] - expected) > 1e-3) {
            correct = false;
            break;
        }
    }

    printf("Result: %s\\n", correct ? "CORRECT" : "INCORRECT");
    printf("Time: %.2f ms\\n", ms);
    printf("Bandwidth: %.2f GB/s\\n", (size * 2 / 1e9) / (ms / 1000.0));
    printf("\\nNote: Zero-copy avoids explicit cudaMemcpy!\\n");
    printf("      Host and device share same physical memory\\n");

    CUDA_CHECK(cudaFreeHost(h_data));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
''',

    'local/phase4/23_multi_kernel_sync.cu': '''#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \\
    do { \\
        cudaError_t err = call; \\
        if (err != cudaSuccess) { \\
            fprintf(stderr, "CUDA error at %s:%d: %s\\n", \\
                    __FILE__, __LINE__, cudaGetErrorString(err)); \\
            exit(EXIT_FAILURE); \\
        } \\
    } while(0)

__global__ void kernel1(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}

__global__ void kernel2(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] + 1.0f;
    }
}

__global__ void kernel3(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = sqrtf(data[idx]);
    }
}

int main() {
    printf("=== Multi-Kernel Synchronization ===\\n\\n");

    int n = 1 << 24;
    size_t size = n * sizeof(float);

    float *h_data = (float*)malloc(size);
    for (int i = 0; i < n; i++) h_data[i] = i + 1.0f;

    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Launch multiple kernels with dependencies
    kernel1<<<blocks, threads>>>(d_data, n);
    CUDA_CHECK(cudaDeviceSynchronize());  // Wait for kernel1

    kernel2<<<blocks, threads>>>(d_data, n);
    CUDA_CHECK(cudaDeviceSynchronize());  // Wait for kernel2

    kernel3<<<blocks, threads>>>(d_data, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    CUDA_CHECK(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));

    // Verify: sqrt((i+1)*2 + 1)
    bool correct = true;
    for (int i = 0; i < 1000; i++) {
        float expected = sqrtf((i + 1.0f) * 2.0f + 1.0f);
        if (abs(h_data[i] - expected) > 1e-3) {
            correct = false;
            printf("Error at %d: got %.3f, expected %.3f\\n", i, h_data[i], expected);
            break;
        }
    }

    printf("Result: %s\\n", correct ? "CORRECT" : "INCORRECT");
    printf("Total time: %.2f ms\\n", ms);
    printf("Three kernels synchronized successfully\\n");

    free(h_data);
    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
''',

    'local/phase6/33_multi_gpu_basic.cu': '''#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \\
    do { \\
        cudaError_t err = call; \\
        if (err != cudaSuccess) { \\
            fprintf(stderr, "CUDA error at %s:%d: %s\\n", \\
                    __FILE__, __LINE__, cudaGetErrorString(err)); \\
            exit(EXIT_FAILURE); \\
        } \\
    } while(0)

__global__ void processKernel(float *data, int n, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int globalIdx = offset + idx;
        data[idx] = sqrtf(globalIdx * 1.0f) * 2.0f;
    }
}

int main() {
    printf("=== Multi-GPU Basic Programming ===\\n\\n");

    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    printf("Found %d CUDA device(s)\\n\\n", deviceCount);

    if (deviceCount < 2) {
        printf("Note: Only %d GPU available.\\n", deviceCount);
        printf("      Demonstrating multi-GPU pattern anyway...\\n\\n");
    }

    int n = 1 << 24;  // Total elements
    int devicesToUse = (deviceCount >= 2) ? 2 : 1;
    int nPerDevice = n / devicesToUse;
    size_t size = nPerDevice * sizeof(float);

    // Allocate on each device
    float **d_data = (float**)malloc(devicesToUse * sizeof(float*));
    float **h_data = (float**)malloc(devicesToUse * sizeof(float*));

    cudaEvent_t *start = (cudaEvent_t*)malloc(devicesToUse * sizeof(cudaEvent_t));
    cudaEvent_t *stop = (cudaEvent_t*)malloc(devicesToUse * sizeof(cudaEvent_t));

    for (int dev = 0; dev < devicesToUse; dev++) {
        CUDA_CHECK(cudaSetDevice(dev));

        h_data[dev] = (float*)malloc(size);
        CUDA_CHECK(cudaMalloc(&d_data[dev], size));

        cudaEventCreate(&start[dev]);
        cudaEventCreate(&stop[dev]);
    }

    // Process on each GPU
    for (int dev = 0; dev < devicesToUse; dev++) {
        CUDA_CHECK(cudaSetDevice(dev));

        int offset = dev * nPerDevice;

        // Initialize
        for (int i = 0; i < nPerDevice; i++) {
            h_data[dev][i] = offset + i;
        }

        CUDA_CHECK(cudaMemcpy(d_data[dev], h_data[dev], size, cudaMemcpyHostToDevice));

        int threads = 256;
        int blocks = (nPerDevice + threads - 1) / threads;

        cudaEventRecord(start[dev]);
        processKernel<<<blocks, threads>>>(d_data[dev], nPerDevice, offset);
        cudaEventRecord(stop[dev]);
    }

    // Wait and collect results
    for (int dev = 0; dev < devicesToUse; dev++) {
        CUDA_CHECK(cudaSetDevice(dev));
        cudaEventSynchronize(stop[dev]);

        float ms;
        cudaEventElapsedTime(&ms, start[dev], stop[dev]);

        CUDA_CHECK(cudaMemcpy(h_data[dev], d_data[dev], size, cudaMemcpyDeviceToHost));

        printf("GPU %d: %.2f ms, %.2f GB/s\\n",
               dev, ms, (size * 2 / 1e9) / (ms / 1000.0));
    }

    // Verify
    bool correct = true;
    for (int dev = 0; dev < devicesToUse; dev++) {
        int offset = dev * nPerDevice;
        for (int i = 0; i < 100; i++) {
            float expected = sqrtf((offset + i) * 1.0f) * 2.0f;
            if (abs(h_data[dev][i] - expected) > 1e-3) {
                correct = false;
                break;
            }
        }
    }

    printf("\\nResult: %s\\n", correct ? "CORRECT" : "INCORRECT");

    // Cleanup
    for (int dev = 0; dev < devicesToUse; dev++) {
        CUDA_CHECK(cudaSetDevice(dev));
        free(h_data[dev]);
        cudaFree(d_data[dev]);
        cudaEventDestroy(start[dev]);
        cudaEventDestroy(stop[dev]);
    }

    free(h_data);
    free(d_data);
    free(start);
    free(stop);

    return 0;
}
''',

    'local/phase7/36_profiling_demo.cu': '''#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \\
    do { \\
        cudaError_t err = call; \\
        if (err != cudaSuccess) { \\
            fprintf(stderr, "CUDA error at %s:%d: %s\\n", \\
                    __FILE__, __LINE__, cudaGetErrorString(err)); \\
            exit(EXIT_FAILURE); \\
        } \\
    } while(0)

// Fast kernel
__global__ void fastKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}

// Memory-bound kernel
__global__ void memoryBoundKernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] + input[(idx + 1) % n] + input[(idx + 2) % n];
    }
}

// Compute-bound kernel
__global__ void computeBoundKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        for (int i = 0; i < 100; i++) {
            val = sqrtf(val + 1.0f);
        }
        data[idx] = val;
    }
}

int main() {
    printf("=== Profiling Demo ===\\n\\n");
    printf("Compile with: nvcc -arch=sm_70 -lineinfo 36_profiling_demo.cu\\n");
    printf("Profile with: nvprof ./a.out\\n");
    printf("Or use: nsys profile ./a.out\\n\\n");

    int n = 1 << 20;
    size_t size = n * sizeof(float);

    float *h_data = (float*)malloc(size);
    for (int i = 0; i < n; i++) h_data[i] = i + 1.0f;

    float *d_data, *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_data, size));
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));

    CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input, h_data, size, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Test 1: Fast kernel
    cudaEventRecord(start);
    fastKernel<<<blocks, threads>>>(d_data, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float fastTime;
    cudaEventElapsedTime(&fastTime, start, stop);

    // Test 2: Memory-bound
    cudaEventRecord(start);
    memoryBoundKernel<<<blocks, threads>>>(d_input, d_output, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float memTime;
    cudaEventElapsedTime(&memTime, start, stop);

    // Test 3: Compute-bound
    cudaEventRecord(start);
    computeBoundKernel<<<blocks, threads>>>(d_data, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float compTime;
    cudaEventElapsedTime(&compTime, start, stop);

    printf("Results:\\n");
    printf("  Fast kernel:         %.2f ms\\n", fastTime);
    printf("  Memory-bound kernel: %.2f ms\\n", memTime);
    printf("  Compute-bound kernel: %.2f ms\\n", compTime);
    printf("\\nUse profiler to see detailed metrics!\\n");

    free(h_data);
    cudaFree(d_data);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
''',

    'local/phase7/38_kernel_fusion.cu': '''#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \\
    do { \\
        cudaError_t err = call; \\
        if (err != cudaSuccess) { \\
            fprintf(stderr, "CUDA error at %s:%d: %s\\n", \\
                    __FILE__, __LINE__, cudaGetErrorString(err)); \\
            exit(EXIT_FAILURE); \\
        } \\
    } while(0)

// Separate kernels (not fused)
__global__ void kernel1(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}

__global__ void kernel2(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] + 10.0f;
    }
}

__global__ void kernel3(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = sqrtf(data[idx]);
    }
}

// Fused kernel (all operations in one)
__global__ void fusedKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        val = val * 2.0f;      // Operation 1
        val = val + 10.0f;     // Operation 2
        val = sqrtf(val);      // Operation 3
        data[idx] = val;
    }
}

int main() {
    printf("=== Kernel Fusion ===\\n\\n");

    int n = 1 << 24;
    size_t size = n * sizeof(float);

    float *h_data = (float*)malloc(size);
    for (int i = 0; i < n; i++) h_data[i] = i + 1.0f;

    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Test 1: Separate kernels
    CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

    cudaEventRecord(start);
    kernel1<<<blocks, threads>>>(d_data, n);
    kernel2<<<blocks, threads>>>(d_data, n);
    kernel3<<<blocks, threads>>>(d_data, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float separateTime;
    cudaEventElapsedTime(&separateTime, start, stop);

    // Test 2: Fused kernel
    CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

    cudaEventRecord(start);
    fusedKernel<<<blocks, threads>>>(d_data, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float fusedTime;
    cudaEventElapsedTime(&fusedTime, start, stop);

    CUDA_CHECK(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));

    // Verify
    bool correct = true;
    for (int i = 0; i < 1000; i++) {
        float expected = sqrtf((i + 1.0f) * 2.0f + 10.0f);
        if (abs(h_data[i] - expected) > 1e-3) {
            correct = false;
            break;
        }
    }

    printf("Result: %s\\n", correct ? "CORRECT" : "INCORRECT");
    printf("\\nSeparate kernels: %.2f ms\\n", separateTime);
    printf("Fused kernel:     %.2f ms\\n", fusedTime);
    printf("Speedup:          %.2fx\\n", separateTime / fusedTime);
    printf("\\nFusion reduces:\\n");
    printf("  - Global memory accesses\\n");
    printf("  - Kernel launch overhead\\n");
    printf("  - Device synchronization\\n");

    free(h_data);
    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
''',
}

def main():
    print("="*70)
    print("Fixing 6 remaining .cu files")
    print("="*70)
    print()

    fixed_count = 0
    for filepath, content in FILES.items():
        try:
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"✓ Fixed: {filepath}")
            fixed_count += 1
        except Exception as e:
            print(f"✗ Error: {filepath} - {e}")

    print(f"\\n{fixed_count}/{len(FILES)} files fixed!")

if __name__ == "__main__":
    main()
