#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


__global__ void kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = data[idx] * 2.0f;
}

int main() {
    printf("=== Multi-GPU Basic ===\n\n");

    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    printf("Available GPUs: %d\n\n", deviceCount);

    if (deviceCount < 2) {
        printf("This demo requires 2+ GPUs. Running on single GPU.\n");
        deviceCount = 1;
    }

    const int N = 1 << 20;
    size_t bytes = N * sizeof(float);

    for (int gpu = 0; gpu < deviceCount; gpu++) {
        CUDA_CHECK(cudaSetDevice(gpu));

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, gpu));
        printf("GPU %d: %s\n", gpu, prop.name);

        float *h_data = (float*)malloc(bytes);
        for (int i = 0; i < N; i++) h_data[i] = (float)i;

        float *d_data;
        CUDA_CHECK(cudaMalloc(&d_data, bytes));
        CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

        kernel<<<(N + 255) / 256, 256>>>(d_data, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));
        printf("  Result[0]: %.0f\n", h_data[0]);

        free(h_data);
        cudaFree(d_data);
    }

    printf("\nMulti-GPU programming requires explicit device management\n");
    return 0;
}
