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
    if (idx < n) {
        for (int i = 0; i < 100; i++)
            data[idx] = sqrtf(data[idx] + 1.0f);
    }
}

int main() {
    printf("=== CUDA Streams Basic ===\n\n");
    const int N = 1 << 20;
    const int nStreams = 4;
    size_t bytes = N * sizeof(float);

    float *h_data = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_data[i] = (float)i;

    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));

    cudaStream_t streams[nStreams];
    for (int i = 0; i < nStreams; i++)
        CUDA_CHECK(cudaStreamCreate(&streams[i]));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int chunkSize = N / nStreams;

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < nStreams; i++) {
        int offset = i * chunkSize;
        int size = (i == nStreams - 1) ? (N - offset) : chunkSize;

        CUDA_CHECK(cudaMemcpyAsync(d_data + offset, h_data + offset,
                   size * sizeof(float), cudaMemcpyHostToDevice, streams[i]));

        kernel<<<(size + 255) / 256, 256, 0, streams[i]>>>(d_data + offset, size);

        CUDA_CHECK(cudaMemcpyAsync(h_data + offset, d_data + offset,
                   size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]));
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Streams: %d\n", nStreams);
    printf("Time: %.3f ms\n", ms);
    printf("Streams enable concurrent execution and overlap\n");

    for (int i = 0; i < nStreams; i++)
        cudaStreamDestroy(streams[i]);

    free(h_data);
    cudaFree(d_data);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
