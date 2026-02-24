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


int main() {
    printf("=== Peer-to-Peer Transfer ===\n\n");

    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    printf("GPUs available: %d\n\n", deviceCount);

    if (deviceCount < 2) {
        printf("P2P requires 2+ GPUs\n");
        return 0;
    }

    int canAccessPeer;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccessPeer, 0, 1));
    printf("GPU 0 can access GPU 1: %s\n", canAccessPeer ? "Yes" : "No");

    if (canAccessPeer) {
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaDeviceEnablePeerAccess(1, 0));

        const int N = 1 << 20;
        size_t bytes = N * sizeof(float);

        float *d_data0, *d_data1;
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaMalloc(&d_data0, bytes));

        CUDA_CHECK(cudaSetDevice(1));
        CUDA_CHECK(cudaMalloc(&d_data1, bytes));

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpyPeer(d_data1, 1, d_data0, 0, bytes));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        printf("P2P transfer: %.3f ms, BW: %.2f GB/s\n",
               ms, bytes / (ms * 1e6));

        CUDA_CHECK(cudaSetDevice(0));
        cudaFree(d_data0);
        CUDA_CHECK(cudaSetDevice(1));
        cudaFree(d_data1);

        cudaEventDestroy(start); cudaEventDestroy(stop);
    }

    return 0;
}
