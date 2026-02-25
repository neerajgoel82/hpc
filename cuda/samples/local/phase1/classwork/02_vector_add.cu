// Phase 1: Vector Addition
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
    printf("=== Vector Addition: CPU vs GPU ===\n\n");

    int n = 10000000;  // 10M elements
    size_t size = n * sizeof(float);
    printf("Vector size: %d elements (%.2f MB)\n\n", n, size / 1024.0 / 1024.0);

    // Allocate host memory
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c_cpu = (float*)malloc(size);
    float *h_c_gpu = (float*)malloc(size);

    // Initialize
    for (int i = 0; i < n; i++) {
        h_a[i] = i * 1.0f;
        h_b[i] = i * 2.0f;
    }

    // CPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float cpu_time, gpu_time;

    cudaEventRecord(start);
    vectorAddCPU(h_a, h_b, h_c_cpu, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpu_time, start, stop);

    // GPU computation
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, size));
    CUDA_CHECK(cudaMalloc(&d_b, size));
    CUDA_CHECK(cudaMalloc(&d_c, size));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    cudaEventRecord(start);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    cudaEventRecord(stop);
    CUDA_CHECK(cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost));
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);

    // Verify
    bool correct = true;
    for (int i = 0; i < n; i++) {
        if (h_c_cpu[i] != h_c_gpu[i]) {
            correct = false;
            break;
        }
    }

    printf("Results:\n");
    printf("  CPU Time: %.2f ms\n", cpu_time);
    printf("  GPU Time: %.2f ms\n", gpu_time);
    printf("  Speedup: %.2fx\n", cpu_time / gpu_time);
    printf("  Verification: %s\n", correct ? "PASSED" : "FAILED");

    // Cleanup
    free(h_a); free(h_b); free(h_c_cpu); free(h_c_gpu);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
