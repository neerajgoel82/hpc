// CUDA Streams
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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


__global__ void kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = sqrt(data[idx]) * 2.0f;
    }
}

int main() {
    printf("=== CUDA Streams ===\n\n");

    int nStreams = 4;
    int n = 10000000;
    int streamSize = n / nStreams;
    size_t streamBytes = streamSize * sizeof(float);

    float *h_data;
    CUDA_CHECK(cudaMallocHost(&h_data, n * sizeof(float)));

    for (int i = 0; i < n; i++) h_data[i] = i;

    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(float)));

    cudaStream_t streams[nStreams];
    for (int i = 0; i < nStreams; i++)
        cudaStreamCreate(&streams[i]);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threadsPerBlock = 256;
    int blocksPerStream = (streamSize + threadsPerBlock - 1) / threadsPerBlock;

    cudaEventRecord(start);

    for (int i = 0; i < nStreams; i++) {
        int offset = i * streamSize;
        CUDA_CHECK(cudaMemcpyAsync(d_data + offset, h_data + offset,
                                   streamBytes, cudaMemcpyHostToDevice, streams[i]));
        kernel<<<blocksPerStream, threadsPerBlock, 0, streams[i]>>>(d_data + offset, streamSize);
        CUDA_CHECK(cudaMemcpyAsync(h_data + offset, d_data + offset,
                                   streamBytes, cudaMemcpyDeviceToHost, streams[i]));
    }

    cudaEventRecord(stop);

    for (int i = 0; i < nStreams; i++)
        cudaStreamSynchronize(streams[i]);

    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    printf("Streams: %d\n", nStreams);
    printf("Elements per stream: %d\n", streamSize);
    printf("Time: %.2f ms\n", ms);
    printf("Throughput: %.2f GB/s\n", (n * sizeof(float) * 3 / 1e9) / (ms / 1000.0));

    for (int i = 0; i < nStreams; i++)
        cudaStreamDestroy(streams[i]);

    cudaFreeHost(h_data);
    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
