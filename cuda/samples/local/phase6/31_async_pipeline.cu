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


__global__ void processKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        for (int i = 0; i < 50; i++)
            data[idx] = sqrtf(fabsf(data[idx]) + 1.0f);
    }
}

int main() {
    printf("=== Async Pipeline ===\n\n");
    const int N = 1 << 20;
    const int nChunks = 8;
    size_t bytes = N * sizeof(float);
    size_t chunkBytes = bytes / nChunks;

    float *h_data = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_data[i] = (float)i;

    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    int chunkSize = N / nChunks;
    for (int i = 0; i < nChunks; i++) {
        int offset = i * chunkSize;
        CUDA_CHECK(cudaMemcpyAsync(d_data + offset, h_data + offset,
                   chunkBytes, cudaMemcpyHostToDevice, stream));
        processKernel<<<(chunkSize + 255) / 256, 256, 0, stream>>>(d_data + offset, chunkSize);
        CUDA_CHECK(cudaMemcpyAsync(h_data + offset, d_data + offset,
                   chunkBytes, cudaMemcpyDeviceToHost, stream));
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Pipelined processing with %d chunks\n", nChunks);
    printf("Time: %.3f ms\n", ms);

    cudaStreamDestroy(stream);
    free(h_data);
    cudaFree(d_data);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
