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
    printf("=== Multi-Kernel Synchronization ===\n\n");

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
            printf("Error at %d: got %.3f, expected %.3f\n", i, h_data[i], expected);
            break;
        }
    }

    printf("Result: %s\n", correct ? "CORRECT" : "INCORRECT");
    printf("Total time: %.2f ms\n", ms);
    printf("Three kernels synchronized successfully\n");

    free(h_data);
    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
