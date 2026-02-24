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


__global__ void zerocopyKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = data[idx] * 2.0f;
}

int main() {
    printf("=== Zero-Copy Memory ===\n\n");

    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    if (!prop.canMapHostMemory) {
        printf("Device does not support mapped memory!\n");
        return 1;
    }

    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));

    const int N = 1 << 20;
    size_t bytes = N * sizeof(float);

    float *h_mapped, *d_mapped;
    CUDA_CHECK(cudaHostAlloc(&h_mapped, bytes, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&d_mapped, h_mapped, 0));

    for (int i = 0; i < N; i++) h_mapped[i] = (float)i;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    zerocopyKernel<<<(N + 255) / 256, 256>>>(d_mapped, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Zero-copy time: %.3f ms\n", ms);
    printf("No explicit cudaMemcpy needed, GPU accesses host memory directly\n");
    printf("Good for: single-use data, avoiding copy overhead\n");

    cudaFreeHost(h_mapped);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
