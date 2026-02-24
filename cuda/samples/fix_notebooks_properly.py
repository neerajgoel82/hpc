#!/usr/bin/env python3
"""
Properly fix all CUDA notebooks with topic-specific implementations.
This script replaces generic template code with real algorithm implementations.
"""

import json
import glob
import os

def get_proper_implementation(filename):
    """Generate proper CUDA code based on the notebook topic."""

    topic = os.path.basename(filename).replace('.ipynb', '')

    # Phase 3: Optimization
    if '12_warp_divergence' in filename:
        return '''%%cu
#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \\
    do { \\
        cudaError_t err = call; \\
        if (err != cudaSuccess) { \\
            fprintf(stderr, "CUDA error: %s\\n", cudaGetErrorString(err)); \\
            exit(EXIT_FAILURE); \\
        } \\
    } while(0)

// Divergent kernel - different threads take different paths
__global__ void divergentKernel(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (idx % 2 == 0) {
            // Even threads do more work
            for (int i = 0; i < 100; i++) {
                data[idx] += i;
            }
        } else {
            // Odd threads do less work
            data[idx] += 1;
        }
    }
}

// Non-divergent kernel - all threads in a warp follow same path
__global__ void nonDivergentKernel(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // All threads do the same amount of work
        for (int i = 0; i < 100; i++) {
            data[idx] += i;
        }
    }
}

int main() {
    printf("=== Warp Divergence Demonstration ===\\n\\n");

    int n = 1 << 20;  // 1M elements
    size_t size = n * sizeof(int);

    int *h_data = (int*)malloc(size);
    for (int i = 0; i < n; i++) h_data[i] = 0;

    int *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Test divergent kernel
    CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
    cudaEventRecord(start);
    divergentKernel<<<blocks, threads>>>(d_data, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float divergentTime;
    cudaEventElapsedTime(&divergentTime, start, stop);

    // Test non-divergent kernel
    CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
    cudaEventRecord(start);
    nonDivergentKernel<<<blocks, threads>>>(d_data, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float nonDivergentTime;
    cudaEventElapsedTime(&nonDivergentTime, start, stop);

    printf("Divergent kernel time: %.2f ms\\n", divergentTime);
    printf("Non-divergent kernel time: %.2f ms\\n", nonDivergentTime);
    printf("Performance difference: %.1fx\\n", divergentTime / nonDivergentTime);

    free(h_data);
    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}'''

    elif '13_warp_shuffle' in filename:
        return '''%%cu
#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \\
    do { \\
        cudaError_t err = call; \\
        if (err != cudaSuccess) { \\
            fprintf(stderr, "CUDA error: %s\\n", cudaGetErrorString(err)); \\
            exit(EXIT_FAILURE); \\
        } \\
    } while(0)

// Warp-level reduction using shuffle operations
__global__ void warpReduceKernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;
    int warpId = threadIdx.x / 32;

    float val = (idx < n) ? input[idx] : 0.0f;

    // Warp-level reduction using shuffle
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    // First thread in each warp writes result
    if (lane == 0) {
        output[blockIdx.x * (blockDim.x / 32) + warpId] = val;
    }
}

int main() {
    printf("=== Warp Shuffle Operations ===\\n\\n");

    int n = 1 << 20;
    size_t size = n * sizeof(float);

    float *h_input = (float*)malloc(size);
    for (int i = 0; i < n; i++) h_input[i] = 1.0f;

    float *d_input, *d_output;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    int outputSize = blocks * (threads / 32);

    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, outputSize * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    warpReduceKernel<<<blocks, threads>>>(d_input, d_output, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    float *h_output = (float*)malloc(outputSize * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_output, d_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost));

    float sum = 0;
    for (int i = 0; i < outputSize; i++) sum += h_output[i];

    printf("Sum: %.0f (expected: %d)\\n", sum, n);
    printf("Time: %.2f ms\\n", ms);
    printf("Bandwidth: %.2f GB/s\\n", (size / 1e9) / (ms / 1000.0));

    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}'''

    elif '14_occupancy' in filename or '14_occupancy_tuning' in filename:
        return '''%%cu
#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \\
    do { \\
        cudaError_t err = call; \\
        if (err != cudaSuccess) { \\
            fprintf(stderr, "CUDA error: %s\\n", cudaGetErrorString(err)); \\
            exit(EXIT_FAILURE); \\
        } \\
    } while(0)

// Kernel with different shared memory usage
template<int SMEM_SIZE>
__global__ void occupancyKernel(float *data, int n) {
    __shared__ float smem[SMEM_SIZE];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < n) {
        smem[tid % SMEM_SIZE] = data[idx];
        __syncthreads();
        data[idx] = smem[tid % SMEM_SIZE] * 2.0f;
    }
}

void testOccupancy(int blockSize) {
    int n = 1 << 24;
    size_t size = n * sizeof(float);

    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size));

    int blocks = (n + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    if (blockSize == 128) {
        occupancyKernel<128><<<blocks, blockSize>>>(d_data, n);
    } else if (blockSize == 256) {
        occupancyKernel<256><<<blocks, blockSize>>>(d_data, n);
    } else if (blockSize == 512) {
        occupancyKernel<512><<<blocks, blockSize>>>(d_data, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    printf("Block size %d: %.2f ms, %.2f GB/s\\n",
           blockSize, ms, (size * 2 / 1e9) / (ms / 1000.0));

    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    printf("=== Occupancy Tuning ===\\n\\n");

    printf("Testing different block sizes:\\n");
    testOccupancy(128);
    testOccupancy(256);
    testOccupancy(512);

    return 0;
}'''

    elif '16_prefix_sum' in filename:
        return '''%%cu
#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \\
    do { \\
        cudaError_t err = call; \\
        if (err != cudaSuccess) { \\
            fprintf(stderr, "CUDA error: %s\\n", cudaGetErrorString(err)); \\
            exit(EXIT_FAILURE); \\
        } \\
    } while(0)

// Parallel prefix sum (scan) using Blelloch algorithm
__global__ void scanKernel(float *data, int n) {
    extern __shared__ float temp[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    temp[tid] = (idx < n) ? data[idx] : 0.0f;
    __syncthreads();

    // Up-sweep (reduce) phase
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            temp[index] += temp[index - stride];
        }
        __syncthreads();
    }

    // Down-sweep phase
    if (tid == 0) temp[blockDim.x - 1] = 0;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            float t = temp[index - stride];
            temp[index - stride] = temp[index];
            temp[index] += t;
        }
        __syncthreads();
    }

    // Write results
    if (idx < n) {
        data[idx] = temp[tid];
    }
}

int main() {
    printf("=== Parallel Prefix Sum (Scan) ===\\n\\n");

    int n = 1024;
    size_t size = n * sizeof(float);

    float *h_data = (float*)malloc(size);
    for (int i = 0; i < n; i++) h_data[i] = 1.0f;

    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    scanKernel<<<blocks, threads, threads * sizeof(float)>>>(d_data, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    CUDA_CHECK(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));

    printf("First 10 values: ");
    for (int i = 0; i < 10; i++) printf("%.0f ", h_data[i]);
    printf("\\n");
    printf("Last value: %.0f (expected: %d)\\n", h_data[n-1], n-1);
    printf("Time: %.2f ms\\n", ms);

    free(h_data);
    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}'''

    elif '17_histogram' in filename:
        return '''%%cu
#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \\
    do { \\
        cudaError_t err = call; \\
        if (err != cudaSuccess) { \\
            fprintf(stderr, "CUDA error: %s\\n", cudaGetErrorString(err)); \\
            exit(EXIT_FAILURE); \\
        } \\
    } while(0)

#define NUM_BINS 256

__global__ void histogramKernel(unsigned char *data, int *histogram, int n) {
    __shared__ int smem[NUM_BINS];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize shared memory
    if (tid < NUM_BINS) {
        smem[tid] = 0;
    }
    __syncthreads();

    // Compute histogram in shared memory
    if (idx < n) {
        atomicAdd(&smem[data[idx]], 1);
    }
    __syncthreads();

    // Write to global memory
    if (tid < NUM_BINS) {
        atomicAdd(&histogram[tid], smem[tid]);
    }
}

int main() {
    printf("=== GPU Histogram ===\\n\\n");

    int n = 1 << 20;  // 1M elements
    size_t size = n * sizeof(unsigned char);

    unsigned char *h_data = (unsigned char*)malloc(size);
    for (int i = 0; i < n; i++) h_data[i] = rand() % 256;

    unsigned char *d_data;
    int *d_histogram;
    CUDA_CHECK(cudaMalloc(&d_data, size));
    CUDA_CHECK(cudaMalloc(&d_histogram, NUM_BINS * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_histogram, 0, NUM_BINS * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    histogramKernel<<<blocks, threads>>>(d_data, d_histogram, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    int *h_histogram = (int*)malloc(NUM_BINS * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_histogram, d_histogram, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost));

    int total = 0;
    for (int i = 0; i < NUM_BINS; i++) total += h_histogram[i];

    printf("Total count: %d (expected: %d)\\n", total, n);
    printf("Time: %.2f ms\\n", ms);
    printf("Throughput: %.2f M elements/s\\n", (n / 1e6) / (ms / 1000.0));

    free(h_data);
    free(h_histogram);
    cudaFree(d_data);
    cudaFree(d_histogram);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}'''

    # Phase 4: Advanced Memory
    elif '18_texture' in filename:
        return '''%%cu
#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \\
    do { \\
        cudaError_t err = call; \\
        if (err != cudaSuccess) { \\
            fprintf(stderr, "CUDA error: %s\\n", cudaGetErrorString(err)); \\
            exit(EXIT_FAILURE); \\
        } \\
    } while(0)

texture<float, 2, cudaReadModeElementType> texRef;

__global__ void textureKernel(float *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // Texture memory provides automatic 2D caching and interpolation
        float value = tex2D(texRef, x, y);
        output[y * width + x] = value * 2.0f;
    }
}

int main() {
    printf("=== Texture Memory ===\\n\\n");

    int width = 1024;
    int height = 1024;
    size_t size = width * height * sizeof(float);

    float *h_input = (float*)malloc(size);
    float *h_output = (float*)malloc(size);
    for (int i = 0; i < width * height; i++) h_input[i] = i;

    // Allocate CUDA array for texture
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray *cuArray;
    CUDA_CHECK(cudaMallocArray(&cuArray, &channelDesc, width, height));
    CUDA_CHECK(cudaMemcpyToArray(cuArray, 0, 0, h_input, size, cudaMemcpyHostToDevice));

    // Bind texture
    texRef.addressMode[0] = cudaAddressModeClamp;
    texRef.addressMode[1] = cudaAddressModeClamp;
    texRef.filterMode = cudaFilterModePoint;
    CUDA_CHECK(cudaBindTextureToArray(texRef, cuArray, channelDesc));

    float *d_output;
    CUDA_CHECK(cudaMalloc(&d_output, size));

    dim3 threads(16, 16);
    dim3 blocks((width + 15) / 16, (height + 15) / 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    textureKernel<<<blocks, threads>>>(d_output, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    printf("Matrix size: %dx%d\\n", width, height);
    printf("Time: %.2f ms\\n", ms);
    printf("Bandwidth: %.2f GB/s\\n", (size * 2 / 1e9) / (ms / 1000.0));

    CUDA_CHECK(cudaUnbindTexture(texRef));
    free(h_input);
    free(h_output);
    cudaFreeArray(cuArray);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}'''

    elif '19_constant' in filename:
        return '''%%cu
#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \\
    do { \\
        cudaError_t err = call; \\
        if (err != cudaSuccess) { \\
            fprintf(stderr, "CUDA error: %s\\n", cudaGetErrorString(err)); \\
            exit(EXIT_FAILURE); \\
        } \\
    } while(0)

#define KERNEL_SIZE 9

__constant__ float c_kernel[KERNEL_SIZE];

__global__ void convolutionConstant(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= 4 && idx < n - 4) {
        float sum = 0.0f;
        for (int i = 0; i < KERNEL_SIZE; i++) {
            sum += input[idx - 4 + i] * c_kernel[i];
        }
        output[idx] = sum;
    }
}

__global__ void convolutionGlobal(float *input, float *kernel, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= 4 && idx < n - 4) {
        float sum = 0.0f;
        for (int i = 0; i < KERNEL_SIZE; i++) {
            sum += input[idx - 4 + i] * kernel[i];
        }
        output[idx] = sum;
    }
}

int main() {
    printf("=== Constant Memory ===\\n\\n");

    int n = 1 << 24;
    size_t size = n * sizeof(float);

    float h_kernel[KERNEL_SIZE] = {0.1f, 0.1f, 0.1f, 0.1f, 0.2f, 0.1f, 0.1f, 0.1f, 0.1f};

    float *h_input = (float*)malloc(size);
    for (int i = 0; i < n; i++) h_input[i] = i % 100;

    float *d_input, *d_output, *d_kernel;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));
    CUDA_CHECK(cudaMalloc(&d_kernel, KERNEL_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(c_kernel, h_kernel, KERNEL_SIZE * sizeof(float)));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Test constant memory
    cudaEventRecord(start);
    convolutionConstant<<<blocks, threads>>>(d_input, d_output, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float constTime;
    cudaEventElapsedTime(&constTime, start, stop);

    // Test global memory
    cudaEventRecord(start);
    convolutionGlobal<<<blocks, threads>>>(d_input, d_kernel, d_output, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float globalTime;
    cudaEventElapsedTime(&globalTime, start, stop);

    printf("Constant memory: %.2f ms\\n", constTime);
    printf("Global memory: %.2f ms\\n", globalTime);
    printf("Speedup: %.2fx\\n", globalTime / constTime);

    free(h_input);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}'''

    elif '20_zero_copy' in filename:
        return '''%%cu
#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \\
    do { \\
        cudaError_t err = call; \\
        if (err != cudaSuccess) { \\
            fprintf(stderr, "CUDA error: %s\\n", cudaGetErrorString(err)); \\
            exit(EXIT_FAILURE); \\
        } \\
    } while(0)

__global__ void processKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}

int main() {
    printf("=== Zero-Copy Memory ===\\n\\n");

    int n = 1 << 20;
    size_t size = n * sizeof(float);

    // Allocate zero-copy (mapped) memory
    float *h_data;
    CUDA_CHECK(cudaHostAlloc(&h_data, size, cudaHostAllocMapped));

    for (int i = 0; i < n; i++) h_data[i] = i;

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
    for (int i = 0; i < n && i < 1000; i++) {
        float expected = i * 2.0f + 1.0f;
        if (h_data[i] != expected) {
            correct = false;
            break;
        }
    }

    printf("Result: %s\\n", correct ? "CORRECT" : "INCORRECT");
    printf("Time: %.2f ms\\n", ms);
    printf("Bandwidth: %.2f GB/s\\n", (size * 2 / 1e9) / (ms / 1000.0));
    printf("Note: Zero-copy avoids explicit cudaMemcpy\\n");

    CUDA_CHECK(cudaFreeHost(h_data));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}'''

    elif '21_atomics' in filename:
        return '''%%cu
#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \\
    do { \\
        cudaError_t err = call; \\
        if (err != cudaSuccess) { \\
            fprintf(stderr, "CUDA error: %s\\n", cudaGetErrorString(err)); \\
            exit(EXIT_FAILURE); \\
        } \\
    } while(0)

__global__ void atomicSumKernel(int *sum, int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(sum, data[idx]);
    }
}

__global__ void atomicMaxKernel(int *maxVal, int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicMax(maxVal, data[idx]);
    }
}

int main() {
    printf("=== Atomic Operations ===\\n\\n");

    int n = 1 << 20;
    size_t size = n * sizeof(int);

    int *h_data = (int*)malloc(size);
    for (int i = 0; i < n; i++) h_data[i] = i % 1000;

    int *d_data, *d_sum, *d_max;
    CUDA_CHECK(cudaMalloc(&d_data, size));
    CUDA_CHECK(cudaMalloc(&d_sum, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_max, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_max, 0, sizeof(int)));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Atomic sum
    cudaEventRecord(start);
    atomicSumKernel<<<blocks, threads>>>(d_sum, d_data, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float sumTime;
    cudaEventElapsedTime(&sumTime, start, stop);

    // Atomic max
    cudaEventRecord(start);
    atomicMaxKernel<<<blocks, threads>>>(d_max, d_data, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float maxTime;
    cudaEventElapsedTime(&maxTime, start, stop);

    int h_sum, h_max;
    CUDA_CHECK(cudaMemcpy(&h_sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_max, d_max, sizeof(int), cudaMemcpyDeviceToHost));

    printf("Atomic sum result: %d\\n", h_sum);
    printf("Atomic max result: %d\\n", h_max);
    printf("Sum time: %.2f ms\\n", sumTime);
    printf("Max time: %.2f ms\\n", maxTime);

    free(h_data);
    cudaFree(d_data);
    cudaFree(d_sum);
    cudaFree(d_max);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}'''

    elif '22_cooperative' in filename:
        return '''%%cu
#include <stdio.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define CUDA_CHECK(call) \\
    do { \\
        cudaError_t err = call; \\
        if (err != cudaSuccess) { \\
            fprintf(stderr, "CUDA error: %s\\n", cudaGetErrorString(err)); \\
            exit(EXIT_FAILURE); \\
        } \\
    } while(0)

__global__ void cooperativeKernel(int *data, int n) {
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();

    int idx = block.group_index().x * block.group_dim().x + block.thread_rank();

    if (idx < n) {
        data[idx] += 1;
    }

    // Grid-wide synchronization
    grid.sync();

    if (idx < n) {
        data[idx] *= 2;
    }
}

int main() {
    printf("=== Cooperative Groups ===\\n\\n");

    int n = 1 << 20;
    size_t size = n * sizeof(int);

    int *h_data = (int*)malloc(size);
    for (int i = 0; i < n; i++) h_data[i] = i;

    int *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    void *kernelArgs[] = {&d_data, &n};

    cudaEventRecord(start);
    CUDA_CHECK(cudaLaunchCooperativeKernel((void*)cooperativeKernel,
                                          dim3(blocks), dim3(threads),
                                          kernelArgs, 0, 0));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    CUDA_CHECK(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));

    // Verify: each element should be (i + 1) * 2
    bool correct = true;
    for (int i = 0; i < n && i < 1000; i++) {
        if (h_data[i] != (i + 1) * 2) {
            correct = false;
            break;
        }
    }

    printf("Result: %s\\n", correct ? "CORRECT" : "INCORRECT");
    printf("Time: %.2f ms\\n", ms);
    printf("Grid-wide synchronization completed\\n");

    free(h_data);
    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}'''

    elif '23_multi_kernel' in filename:
        return '''%%cu
#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \\
    do { \\
        cudaError_t err = call; \\
        if (err != cudaSuccess) { \\
            fprintf(stderr, "CUDA error: %s\\n", cudaGetErrorString(err)); \\
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
        data[idx] = data[idx] / 2.0f;
    }
}

int main() {
    printf("=== Multi-Kernel Synchronization ===\\n\\n");

    int n = 1 << 24;
    size_t size = n * sizeof(float);

    float *h_data = (float*)malloc(size);
    for (int i = 0; i < n; i++) h_data[i] = i;

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

    // Verify: (i * 2 + 1) / 2
    bool correct = true;
    for (int i = 0; i < 1000; i++) {
        float expected = (i * 2.0f + 1.0f) / 2.0f;
        if (abs(h_data[i] - expected) > 0.001f) {
            correct = false;
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
}'''

    # Continue with Phase 5-9 implementations in next part...
    else:
        # Return None if we don't have a specific implementation yet
        return None

# This is part 1 - focusing on Phase 3-4
# Will create part 2 for Phase 5-9

def main():
    print("Fixing CUDA notebooks with proper implementations...")
    print("Part 1: Phases 3-4 (Optimization and Advanced Memory)")

    notebooks_dir = "colab/notebooks"
    fixed_count = 0

    for filename in glob.glob(f"{notebooks_dir}/phase*/*.ipynb"):
        code = get_proper_implementation(filename)

        if code is None:
            continue

        try:
            with open(filename, 'r') as f:
                notebook = json.load(f)

            # Find the main example cell (usually cell-3)
            for cell in notebook['cells']:
                if cell.get('id') == 'cell-3' and '%%cu' in ''.join(cell.get('source', [])):
                    cell['source'] = code.split('\n')
                    fixed_count += 1
                    print(f"✓ Fixed: {os.path.basename(filename)}")
                    break

            with open(filename, 'w') as f:
                json.dump(notebook, f, indent=1)

        except Exception as e:
            print(f"✗ Error fixing {filename}: {e}")

    print(f"\nFixed {fixed_count} notebooks in Part 1")

if __name__ == "__main__":
    main()
