#!/usr/bin/env python3
"""
Part 2: Fix notebooks for Phases 5-9 with proper implementations.
"""

import json
import glob
import os

def get_implementation_part2(filename):
    """Generate proper CUDA code for Phases 5-9."""

    # Phase 5: Advanced Algorithms
    if '24_gemm' in filename:
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

#define TILE_SIZE 16

// Tiled matrix multiplication using shared memory
__global__ void gemmTiled(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile into shared memory
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t * TILE_SIZE + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial product
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

int main() {
    printf("=== Optimized GEMM (Tiled) ===\\n\\n");

    int M = 1024, N = 1024, K = 1024;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);

    for (int i = 0; i < M * K; i++) h_A[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < K * N; i++) h_B[i] = rand() / (float)RAND_MAX;

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    gemmTiled<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    float gflops = (2.0f * M * N * K) / (ms / 1000.0f) / 1e9f;

    printf("Matrix size: %dx%dx%d\\n", M, K, N);
    printf("Time: %.2f ms\\n", ms);
    printf("Performance: %.2f GFLOPS\\n", gflops);

    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}'''

    elif '25_cublas' in filename:
        return '''%%cu
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call) \\
    do { \\
        cudaError_t err = call; \\
        if (err != cudaSuccess) { \\
            fprintf(stderr, "CUDA error: %s\\n", cudaGetErrorString(err)); \\
            exit(EXIT_FAILURE); \\
        } \\
    } while(0)

int main() {
    printf("=== cuBLAS Matrix Multiplication ===\\n\\n");

    int M = 2048, N = 2048, K = 2048;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);

    for (int i = 0; i < M * K; i++) h_A[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < K * N; i++) h_B[i] = rand() / (float)RAND_MAX;

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f, beta = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha,
                d_B, N, d_A, K,
                &beta, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    float gflops = (2.0f * M * N * K) / (ms / 1000.0f) / 1e9f;

    printf("Matrix size: %dx%dx%d\\n", M, K, N);
    printf("cuBLAS time: %.2f ms\\n", ms);
    printf("Performance: %.2f GFLOPS\\n", gflops);

    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cublasDestroy(handle);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}'''

    elif '26_matrix_transpose' in filename:
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

#define TILE_DIM 32

__global__ void transposeCoalesced(float *out, float *in, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 to avoid bank conflicts

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Read from input (coalesced)
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = in[y * width + x];
    }

    __syncthreads();

    // Write to output (coalesced after transpose)
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    if (x < height && y < width) {
        out[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}

int main() {
    printf("=== Optimized Matrix Transpose ===\\n\\n");

    int width = 4096;
    int height = 4096;
    size_t size = width * height * sizeof(float);

    float *h_in = (float*)malloc(size);
    float *h_out = (float*)malloc(size);
    for (int i = 0; i < width * height; i++) h_in[i] = i;

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, size));
    CUDA_CHECK(cudaMalloc(&d_out, size));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));

    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 blocks((width + TILE_DIM - 1) / TILE_DIM,
                (height + TILE_DIM - 1) / TILE_DIM);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    transposeCoalesced<<<blocks, threads>>>(d_out, d_in, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    CUDA_CHECK(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));

    // Verify
    bool correct = true;
    for (int i = 0; i < 100; i++) {
        int row = i / width;
        int col = i % width;
        if (h_out[col * height + row] != h_in[i]) {
            correct = false;
            break;
        }
    }

    printf("Matrix: %dx%d\\n", width, height);
    printf("Result: %s\\n", correct ? "CORRECT" : "INCORRECT");
    printf("Time: %.2f ms\\n", ms);
    printf("Bandwidth: %.2f GB/s\\n", (size * 2 / 1e9) / (ms / 1000.0));

    free(h_in); free(h_out);
    cudaFree(d_in); cudaFree(d_out);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}'''

    elif '27_bitonic' in filename:
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

__global__ void bitonicSort(float *data, int j, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ixj = idx ^ j;

    if (ixj > idx) {
        if ((idx & k) == 0) {
            if (data[idx] > data[ixj]) {
                float temp = data[idx];
                data[idx] = data[ixj];
                data[ixj] = temp;
            }
        } else {
            if (data[idx] < data[ixj]) {
                float temp = data[idx];
                data[idx] = data[ixj];
                data[ixj] = temp;
            }
        }
    }
}

int main() {
    printf("=== Bitonic Sort ===\\n\\n");

    int n = 1 << 20;  // Must be power of 2
    size_t size = n * sizeof(float);

    float *h_data = (float*)malloc(size);
    for (int i = 0; i < n; i++) h_data[i] = rand() / (float)RAND_MAX;

    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = n / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int k = 2; k <= n; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            bitonicSort<<<blocks, threads>>>(d_data, j, k);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    CUDA_CHECK(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));

    // Verify
    bool sorted = true;
    for (int i = 0; i < n - 1; i++) {
        if (h_data[i] > h_data[i + 1]) {
            sorted = false;
            break;
        }
    }

    printf("Array size: %d\\n", n);
    printf("Result: %s\\n", sorted ? "SORTED" : "NOT SORTED");
    printf("Time: %.2f ms\\n", ms);

    free(h_data);
    cudaFree(d_data);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}'''

    elif '28_radix' in filename:
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

__global__ void radixSortPass(unsigned int *input, unsigned int *output,
                               int n, int bit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        unsigned int value = input[idx];
        unsigned int bitValue = (value >> bit) & 1;

        // Simple radix sort pass (simplified version)
        if (bitValue == 0) {
            atomicAdd(&output[0], 1);  // Count zeros
        }
    }
    __syncthreads();

    // Full implementation would require prefix sum and rearrangement
}

int main() {
    printf("=== Radix Sort (Simplified) ===\\n\\n");
    printf("Note: Full radix sort requires multiple passes with prefix sums\\n\\n");

    int n = 1 << 16;
    size_t size = n * sizeof(unsigned int);

    unsigned int *h_data = (unsigned int*)malloc(size);
    for (int i = 0; i < n; i++) h_data[i] = rand();

    unsigned int *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));
    CUDA_CHECK(cudaMemcpy(d_input, h_data, size, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Demonstrate one pass
    for (int bit = 0; bit < 32; bit += 8) {
        radixSortPass<<<blocks, threads>>>(d_input, d_output, n, bit);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    printf("Array size: %d\\n", n);
    printf("Time: %.2f ms\\n", ms);
    printf("Throughput: %.2f M elements/s\\n", (n / 1e6) / (ms / 1000.0));

    free(h_data);
    cudaFree(d_input); cudaFree(d_output);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}'''

    elif '29_thrust' in filename:
        return '''%%cu
#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <cuda_runtime.h>

struct square_functor {
    __host__ __device__
    float operator()(float x) const {
        return x * x;
    }
};

int main() {
    printf("=== Thrust Library Examples ===\\n\\n");

    int n = 1 << 20;

    // Create device vector
    thrust::device_vector<float> d_vec(n);

    // Fill with data
    for (int i = 0; i < n; i++) {
        d_vec[i] = rand() / (float)RAND_MAX;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Test 1: Sort
    cudaEventRecord(start);
    thrust::sort(d_vec.begin(), d_vec.end());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float sortTime;
    cudaEventElapsedTime(&sortTime, start, stop);

    // Test 2: Reduce (sum)
    cudaEventRecord(start);
    float sum = thrust::reduce(d_vec.begin(), d_vec.end(), 0.0f);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float reduceTime;
    cudaEventElapsedTime(&reduceTime, start, stop);

    // Test 3: Transform (square)
    thrust::device_vector<float> d_result(n);
    cudaEventRecord(start);
    thrust::transform(d_vec.begin(), d_vec.end(), d_result.begin(), square_functor());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float transformTime;
    cudaEventElapsedTime(&transformTime, start, stop);

    printf("Vector size: %d\\n", n);
    printf("Sort time: %.2f ms\\n", sortTime);
    printf("Reduce time: %.2f ms\\n", reduceTime);
    printf("Transform time: %.2f ms\\n", transformTime);
    printf("Sum: %.2f\\n", sum);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}'''

    # Phase 6: Streams & Concurrency
    elif '30_streams' in filename:
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

__global__ void kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        for (int i = 0; i < 1000; i++) {
            data[idx] = data[idx] * 1.01f;
        }
    }
}

int main() {
    printf("=== CUDA Streams ===\\n\\n");

    int n = 1 << 20;
    int numStreams = 4;
    int streamSize = n / numStreams;
    size_t streamBytes = streamSize * sizeof(float);

    float *h_data;
    CUDA_CHECK(cudaHostAlloc(&h_data, n * sizeof(float), cudaHostAllocDefault));
    for (int i = 0; i < n; i++) h_data[i] = i;

    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(float)));

    cudaStream_t streams[numStreams];
    for (int i = 0; i < numStreams; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    int threads = 256;
    int blocks = (streamSize + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Without streams
    cudaEventRecord(start);
    CUDA_CHECK(cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice));
    kernel<<<n / threads, threads>>>(d_data, n);
    CUDA_CHECK(cudaMemcpy(h_data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float noStreamTime;
    cudaEventElapsedTime(&noStreamTime, start, stop);

    // With streams
    cudaEventRecord(start);
    for (int i = 0; i < numStreams; i++) {
        int offset = i * streamSize;
        CUDA_CHECK(cudaMemcpyAsync(&d_data[offset], &h_data[offset],
                                   streamBytes, cudaMemcpyHostToDevice, streams[i]));
        kernel<<<blocks, threads, 0, streams[i]>>>(&d_data[offset], streamSize);
        CUDA_CHECK(cudaMemcpyAsync(&h_data[offset], &d_data[offset],
                                   streamBytes, cudaMemcpyDeviceToHost, streams[i]));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float streamTime;
    cudaEventElapsedTime(&streamTime, start, stop);

    printf("Without streams: %.2f ms\\n", noStreamTime);
    printf("With %d streams: %.2f ms\\n", numStreams, streamTime);
    printf("Speedup: %.2fx\\n", noStreamTime / streamTime);

    for (int i = 0; i < numStreams; i++) {
        cudaStreamDestroy(streams[i]);
    }
    cudaFreeHost(h_data);
    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}'''

    elif '31_async' in filename:
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
        data[idx] = sqrtf(data[idx]) * 2.0f;
    }
}

int main() {
    printf("=== Asynchronous Pipeline ===\\n\\n");

    int n = 1 << 22;
    int numChunks = 4;
    int chunkSize = n / numChunks;
    size_t chunkBytes = chunkSize * sizeof(float);

    float *h_data;
    CUDA_CHECK(cudaHostAlloc(&h_data, n * sizeof(float), cudaHostAllocDefault));
    for (int i = 0; i < n; i++) h_data[i] = i + 1.0f;

    float *d_data[numChunks];
    cudaStream_t streams[numChunks];

    for (int i = 0; i < numChunks; i++) {
        CUDA_CHECK(cudaMalloc(&d_data[i], chunkBytes));
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    int threads = 256;
    int blocks = (chunkSize + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Overlap computation and transfers
    for (int i = 0; i < numChunks; i++) {
        int offset = i * chunkSize;

        // H2D transfer
        CUDA_CHECK(cudaMemcpyAsync(d_data[i], &h_data[offset],
                                   chunkBytes, cudaMemcpyHostToDevice, streams[i]));

        // Kernel execution
        processKernel<<<blocks, threads, 0, streams[i]>>>(d_data[i], chunkSize);

        // D2H transfer
        CUDA_CHECK(cudaMemcpyAsync(&h_data[offset], d_data[i],
                                   chunkBytes, cudaMemcpyDeviceToHost, streams[i]));
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    printf("Processed %d elements in %.2f ms\\n", n, ms);
    printf("Throughput: %.2f M elements/s\\n", (n / 1e6) / (ms / 1000.0));
    printf("Pipeline stages: %d\\n", numChunks);

    for (int i = 0; i < numChunks; i++) {
        cudaStreamDestroy(streams[i]);
        cudaFree(d_data[i]);
    }
    cudaFreeHost(h_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}'''

    elif '32_events' in filename:
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
        data[idx] = sqrtf(data[idx]);
    }
}

__global__ void kernel2(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * data[idx];
    }
}

int main() {
    printf("=== CUDA Events and Timing ===\\n\\n");

    int n = 1 << 24;
    size_t size = n * sizeof(float);

    float *h_data = (float*)malloc(size);
    for (int i = 0; i < n; i++) h_data[i] = i + 1.0f;

    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    cudaEvent_t start, stop, mid1, mid2;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&mid1);
    cudaEventCreate(&mid2);

    // Record events at different points
    cudaEventRecord(start);

    CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
    cudaEventRecord(mid1);

    kernel1<<<blocks, threads>>>(d_data, n);
    cudaEventRecord(mid2);

    kernel2<<<blocks, threads>>>(d_data, n);

    CUDA_CHECK(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float transferTime, kernel1Time, kernel2Time, totalTime;
    cudaEventElapsedTime(&transferTime, start, mid1);
    cudaEventElapsedTime(&kernel1Time, mid1, mid2);
    cudaEventElapsedTime(&kernel2Time, mid2, stop);
    cudaEventElapsedTime(&totalTime, start, stop);

    printf("H2D Transfer: %.2f ms\\n", transferTime);
    printf("Kernel 1 (sqrt): %.2f ms\\n", kernel1Time);
    printf("Kernel 2 (square): %.2f ms\\n", kernel2Time);
    printf("Total time: %.2f ms\\n", totalTime);

    free(h_data);
    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(mid1);
    cudaEventDestroy(mid2);

    return 0;
}'''

    # Continue in Part 3 for remaining notebooks...
    else:
        return None

def main():
    print("Fixing CUDA notebooks - Part 2...")
    print("Phase 5-6 implementations")

    notebooks_dir = "colab/notebooks"
    fixed_count = 0

    for filename in glob.glob(f"{notebooks_dir}/phase*/*.ipynb"):
        code = get_implementation_part2(filename)

        if code is None:
            continue

        try:
            with open(filename, 'r') as f:
                notebook = json.load(f)

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

    print(f"\nFixed {fixed_count} notebooks in Part 2")

if __name__ == "__main__":
    main()
