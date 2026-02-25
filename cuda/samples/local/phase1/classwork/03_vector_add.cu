#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

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
    printf("=== Vector Addition ===\n\n");

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

    printf("Vector size: %d elements\n", n);
    printf("GPU time: %.2f ms\n", gpuTime);
    printf("CPU time: %.2f ms\n", cpuTime);
    printf("Speedup: %.2fx\n", cpuTime / gpuTime);
    printf("Result: %s\n", correct ? "CORRECT" : "INCORRECT");

    free(h_a); free(h_b); free(h_c); free(h_c_cpu);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
