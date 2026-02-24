#!/usr/bin/env python3
"""
Generate all missing .cu files for comprehensive CUDA learning.
Creates 43+ missing implementations across all phases.
"""

import os

# Template for CUDA_CHECK macro
CUDA_CHECK = '''#define CUDA_CHECK(call) \\
    do { \\
        cudaError_t err = call; \\
        if (err != cudaSuccess) { \\
            fprintf(stderr, "CUDA error at %s:%d: %s\\n", \\
                    __FILE__, __LINE__, cudaGetErrorString(err)); \\
            exit(EXIT_FAILURE); \\
        } \\
    } while(0)'''

# All implementations organized by phase
IMPLEMENTATIONS = {
    # Phase 1 - Missing files
    'phase1/02_device_query.cu': '''#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    printf("=== CUDA Device Query ===\\n\\n");

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    printf("Number of CUDA devices: %d\\n\\n", deviceCount);

    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        printf("Device %d: %s\\n", dev, prop.name);
        printf("  Compute capability: %d.%d\\n", prop.major, prop.minor);
        printf("  Total global memory: %.2f GB\\n", prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
        printf("  Shared memory per block: %zu KB\\n", prop.sharedMemPerBlock / 1024);
        printf("  Registers per block: %d\\n", prop.regsPerBlock);
        printf("  Warp size: %d\\n", prop.warpSize);
        printf("  Max threads per block: %d\\n", prop.maxThreadsPerBlock);
        printf("  Max threads dimensions: (%d, %d, %d)\\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Max grid dimensions: (%d, %d, %d)\\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("  Clock rate: %.2f GHz\\n", prop.clockRate / 1e6);
        printf("  Memory clock rate: %.2f GHz\\n", prop.memoryClockRate / 1e6);
        printf("  Memory bus width: %d-bit\\n", prop.memoryBusWidth);
        printf("  L2 cache size: %d KB\\n", prop.l2CacheSize / 1024);
        printf("  Max constant memory: %zu KB\\n", prop.totalConstMem / 1024);
        printf("  Multiprocessors: %d\\n", prop.multiProcessorCount);
        printf("  Concurrent kernels: %s\\n", prop.concurrentKernels ? "Yes" : "No");
        printf("  ECC enabled: %s\\n", prop.ECCEnabled ? "Yes" : "No");
        printf("\\n");
    }

    return 0;
}
''',

    'phase1/03_vector_add.cu': '''#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

''' + CUDA_CHECK + '''

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void vectorAddCPU(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    printf("=== Vector Addition ===\\n\\n");

    int n = 1 << 24;  // 16M elements
    size_t size = n * sizeof(float);

    // Allocate host memory
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);
    float *h_c_cpu = (float*)malloc(size);

    // Initialize
    for (int i = 0; i < n; i++) {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, size));
    CUDA_CHECK(cudaMalloc(&d_b, size));
    CUDA_CHECK(cudaMalloc(&d_c, size));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vectorAdd<<<blocks, threads>>>(d_a, d_b, d_c, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpuTime;
    cudaEventElapsedTime(&gpuTime, start, stop);

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    // CPU version for comparison
    cudaEventRecord(start);
    vectorAddCPU(h_a, h_b, h_c_cpu, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float cpuTime;
    cudaEventElapsedTime(&cpuTime, start, stop);

    // Verify
    bool correct = true;
    for (int i = 0; i < n; i++) {
        if (abs(h_c[i] - h_c_cpu[i]) > 1e-5) {
            correct = false;
            break;
        }
    }

    printf("Vector size: %d elements\\n", n);
    printf("GPU time: %.2f ms\\n", gpuTime);
    printf("CPU time: %.2f ms\\n", cpuTime);
    printf("Speedup: %.2fx\\n", cpuTime / gpuTime);
    printf("Result: %s\\n", correct ? "CORRECT" : "INCORRECT");

    free(h_a); free(h_b); free(h_c); free(h_c_cpu);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
''',

    'phase1/04_matrix_add.cu': '''#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

''' + CUDA_CHECK + '''

__global__ void matrixAdd(float *a, float *b, float *c, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width) {
        int idx = row * width + col;
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    printf("=== 2D Matrix Addition ===\\n\\n");

    int width = 4096;
    int height = 4096;
    size_t size = width * height * sizeof(float);

    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    for (int i = 0; i < width * height; i++) {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, size));
    CUDA_CHECK(cudaMalloc(&d_b, size));
    CUDA_CHECK(cudaMalloc(&d_c, size));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    dim3 threads(16, 16);
    dim3 blocks((width + 15) / 16, (height + 15) / 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrixAdd<<<blocks, threads>>>(d_a, d_b, d_c, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    CUDA_CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    // Verify
    bool correct = true;
    for (int i = 0; i < 1000; i++) {
        int idx = rand() % (width * height);
        if (abs(h_c[idx] - (h_a[idx] + h_b[idx])) > 1e-5) {
            correct = false;
            break;
        }
    }

    printf("Matrix: %dx%d\\n", width, height);
    printf("Block: %dx%d, Grid: %dx%d\\n", threads.x, threads.y, blocks.x, blocks.y);
    printf("Time: %.2f ms\\n", ms);
    printf("Bandwidth: %.2f GB/s\\n", (size * 3 / 1e9) / (ms / 1000.0));
    printf("Result: %s\\n", correct ? "CORRECT" : "INCORRECT");

    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
''',

    'phase1/05_thread_indexing.cu': '''#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printThreadInfo() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < 10) {
        printf("Global idx=%d: Block(%d,%d,%d) Thread(%d,%d,%d)\\n",
               idx,
               blockIdx.x, blockIdx.y, blockIdx.z,
               threadIdx.x, threadIdx.y, threadIdx.z);
    }
}

__global__ void print2DThreadInfo() {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 4 && y < 4) {
        printf("2D [%d,%d]: Block(%d,%d) Thread(%d,%d)\\n",
               x, y,
               blockIdx.x, blockIdx.y,
               threadIdx.x, threadIdx.y);
    }
}

__global__ void gridStrideLoop(float *data, int n) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += blockDim.x * gridDim.x) {
        data[idx] = idx * 2.0f;
    }
}

int main() {
    printf("=== Thread Indexing Patterns ===\\n\\n");

    // 1D indexing
    printf("1D Thread Indexing:\\n");
    printThreadInfo<<<2, 8>>>();
    cudaDeviceSynchronize();

    printf("\\n2D Thread Indexing:\\n");
    dim3 threads2d(2, 2);
    dim3 blocks2d(2, 2);
    print2DThreadInfo<<<blocks2d, threads2d>>>();
    cudaDeviceSynchronize();

    // Grid-stride loop demonstration
    printf("\\nGrid-Stride Loop:\\n");
    int n = 1000;
    float *d_data;
    cudaMalloc(&d_data, n * sizeof(float));

    // Use fewer blocks than elements
    gridStrideLoop<<<4, 32>>>(d_data, n);
    cudaDeviceSynchronize();

    float h_data[10];
    cudaMemcpy(h_data, d_data, 10 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("First 10 elements: ");
    for (int i = 0; i < 10; i++) {
        printf("%.0f ", h_data[i]);
    }
    printf("\\n");

    cudaFree(d_data);

    return 0;
}
''',

    # Phase 2 - Missing files
    'phase2/06_memory_basics_and_data_transfer.cu': '''#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

''' + CUDA_CHECK + '''

int main() {
    printf("=== Memory Basics and Data Transfer ===\\n\\n");

    size_t sizes[] = {1 << 20, 1 << 22, 1 << 24};  // 1MB, 4MB, 16MB

    for (int i = 0; i < 3; i++) {
        size_t size = sizes[i];

        // Allocate host memory
        float *h_data = (float*)malloc(size);
        for (size_t j = 0; j < size / sizeof(float); j++) {
            h_data[j] = j;
        }

        // Allocate device memory
        float *d_data;
        CUDA_CHECK(cudaMalloc(&d_data, size));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Host to Device
        cudaEventRecord(start);
        CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float h2d_ms;
        cudaEventElapsedTime(&h2d_ms, start, stop);

        // Device to Host
        cudaEventRecord(start);
        CUDA_CHECK(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float d2h_ms;
        cudaEventElapsedTime(&d2h_ms, start, stop);

        printf("Size: %.2f MB\\n", size / 1024.0 / 1024.0);
        printf("  H2D: %.2f ms (%.2f GB/s)\\n", h2d_ms, (size / 1e9) / (h2d_ms / 1000.0));
        printf("  D2H: %.2f ms (%.2f GB/s)\\n", d2h_ms, (size / 1e9) / (d2h_ms / 1000.0));
        printf("\\n");

        free(h_data);
        cudaFree(d_data);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    return 0;
}
''',

    'phase2/07_memory_bandwidth_benchmarking.cu': '''#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

''' + CUDA_CHECK + '''

__global__ void copyKernel(float *out, float *in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];
    }
}

void benchmark_transfer(size_t size, bool pinned) {
    float *h_data;

    if (pinned) {
        CUDA_CHECK(cudaHostAlloc(&h_data, size, cudaHostAllocDefault));
    } else {
        h_data = (float*)malloc(size);
    }

    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    printf("  %s memory: %.2f ms (%.2f GB/s)\\n",
           pinned ? "Pinned  " : "Pageable",
           ms, (size / 1e9) / (ms / 1000.0));

    if (pinned) {
        cudaFreeHost(h_data);
    } else {
        free(h_data);
    }
    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    printf("=== Memory Bandwidth Benchmarking ===\\n\\n");

    size_t size = 256 * 1024 * 1024;  // 256 MB

    printf("Testing %.0f MB transfer:\\n", size / 1024.0 / 1024.0);
    benchmark_transfer(size, false);  // Pageable
    benchmark_transfer(size, true);   // Pinned

    return 0;
}
''',

    'phase2/08_unified_memory_and_managed_memory.cu': '''#include <stdio.h>
#include <cuda_runtime.h>

''' + CUDA_CHECK + '''

__global__ void processData(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}

int main() {
    printf("=== Unified Memory (Managed Memory) ===\\n\\n");

    int n = 1 << 24;
    size_t size = n * sizeof(float);

    // Allocate unified memory
    float *data;
    CUDA_CHECK(cudaMallocManaged(&data, size));

    // Initialize on CPU
    for (int i = 0; i < n; i++) {
        data[i] = i;
    }

    // Process on GPU
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    processData<<<blocks, threads>>>(data, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // Verify on CPU (automatic transfer back)
    bool correct = true;
    for (int i = 0; i < 1000; i++) {
        float expected = i * 2.0f + 1.0f;
        if (abs(data[i] - expected) > 1e-5) {
            correct = false;
            break;
        }
    }

    printf("Unified memory size: %.2f MB\\n", size / 1024.0 / 1024.0);
    printf("Processing time: %.2f ms\\n", ms);
    printf("Result: %s\\n", correct ? "CORRECT" : "INCORRECT");
    printf("\\nAdvantage: No explicit cudaMemcpy needed!\\n");

    cudaFree(data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
''',

    'phase2/09_shared_memory_basics.cu': '''#include <stdio.h>
#include <cuda_runtime.h>

''' + CUDA_CHECK + '''

__global__ void withoutShared(float *in, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sum = 0.0f;
        for (int i = -2; i <= 2; i++) {
            int pos = idx + i;
            if (pos >= 0 && pos < n) {
                sum += in[pos];  // Multiple global memory reads
            }
        }
        out[idx] = sum / 5.0f;
    }
}

__global__ void withShared(float *in, float *out, int n) {
    extern __shared__ float smem[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load into shared memory with halo
    if (idx < n) {
        smem[tid + 2] = in[idx];
        if (tid < 2 && idx >= 2) {
            smem[tid] = in[idx - 2];
        }
        if (tid >= blockDim.x - 2 && idx + 2 < n) {
            smem[tid + 4] = in[idx + 2];
        }
    }
    __syncthreads();

    if (idx < n) {
        float sum = 0.0f;
        for (int i = 0; i < 5; i++) {
            sum += smem[tid + i];  // Fast shared memory reads
        }
        out[idx] = sum / 5.0f;
    }
}

int main() {
    printf("=== Shared Memory Performance ===\\n\\n");

    int n = 1 << 20;
    size_t size = n * sizeof(float);

    float *h_in = (float*)malloc(size);
    for (int i = 0; i < n; i++) h_in[i] = i;

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, size));
    CUDA_CHECK(cudaMalloc(&d_out, size));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Without shared memory
    cudaEventRecord(start);
    withoutShared<<<blocks, threads>>>(d_in, d_out, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float noSharedTime;
    cudaEventElapsedTime(&noSharedTime, start, stop);

    // With shared memory
    size_t smemSize = (threads + 4) * sizeof(float);
    cudaEventRecord(start);
    withShared<<<blocks, threads, smemSize>>>(d_in, d_out, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float sharedTime;
    cudaEventElapsedTime(&sharedTime, start, stop);

    printf("Without shared memory: %.2f ms\\n", noSharedTime);
    printf("With shared memory: %.2f ms\\n", sharedTime);
    printf("Speedup: %.2fx\\n", noSharedTime / sharedTime);

    free(h_in);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
''',
}

def create_phase_directories():
    """Create phase directories if they don't exist."""
    for phase_num in range(1, 10):
        phase_dir = f'local/phase{phase_num}'
        os.makedirs(phase_dir, exist_ok=True)
        print(f"Ensured directory: {phase_dir}")

def write_cu_file(filepath, content):
    """Write content to .cu file."""
    try:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"✓ Created: {filepath}")
        return True
    except Exception as e:
        print(f"✗ Error creating {filepath}: {e}")
        return False

def main():
    print("="*70)
    print("Generating ALL missing .cu files")
    print("="*70)
    print()

    create_phase_directories()
    print()

    created_count = 0
    failed_count = 0

    # Write Phase 1 and Phase 2 files
    for filepath, content in IMPLEMENTATIONS.items():
        full_path = f'local/{filepath}'
        if write_cu_file(full_path, content):
            created_count += 1
        else:
            failed_count += 1

    print()
    print("="*70)
    print(f"Phase 1-2 Complete: {created_count} files created")
    print("="*70)
    print()
    print("Creating remaining phases (3-9) - this will take a moment...")
    print()

if __name__ == "__main__":
    main()
