// Phase 1: Hello World from GPU
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


__global__ void helloKernel() {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    printf("Hello from GPU thread %d!\n", tid);
}

int main() {
    printf("=== CUDA Hello World ===\n\n");

    // Query device
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("Running on: %s\n\n", prop.name);

    // Launch kernel
    printf("Launching kernel with 2 blocks, 4 threads each...\n");
    helloKernel<<<2, 4>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("\nKernel completed successfully!\n");
    return 0;
}
