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

__global__ void processKernel(float *data, int n, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int globalIdx = offset + idx;
        data[idx] = sqrtf(globalIdx * 1.0f) * 2.0f;
    }
}

int main() {
    printf("=== Multi-GPU Basic Programming ===\n\n");

    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    printf("Found %d CUDA device(s)\n\n", deviceCount);

    if (deviceCount < 2) {
        printf("Note: Only %d GPU available.\n", deviceCount);
        printf("      Demonstrating multi-GPU pattern anyway...\n\n");
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

        printf("GPU %d: %.2f ms, %.2f GB/s\n",
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

    printf("\nResult: %s\n", correct ? "CORRECT" : "INCORRECT");

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
