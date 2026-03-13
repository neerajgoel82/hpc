#include <stdio.h>
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

__global__ void processData(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}

int main() {
    printf("=== Unified Memory (Managed Memory) ===\n\n");

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

    printf("Unified memory size: %.2f MB\n", size / 1024.0 / 1024.0);
    printf("Processing time: %.2f ms\n", ms);
    printf("Result: %s\n", correct ? "CORRECT" : "INCORRECT");
    printf("\nAdvantage: No explicit cudaMemcpy needed!\n");

    cudaFree(data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
