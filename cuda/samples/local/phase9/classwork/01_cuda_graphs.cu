// CUDA Graphs
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


__global__ void kernel1(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] *= 2.0f;
}

__global__ void kernel2(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] += 1.0f;
}

int main() {
    printf("=== CUDA Graphs ===\n\n");

    int n = 10000000;
    size_t size = n * sizeof(float);

    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size));

    int blocks = (n + 255) / 256;
    int threads = 256;

    // Create graph
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    cudaStream_t stream;

    cudaStreamCreate(&stream);

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    kernel1<<<blocks, threads, 0, stream>>>(d_data, n);
    kernel2<<<blocks, threads, 0, stream>>>(d_data, n);
    cudaStreamEndCapture(stream, &graph);

    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

    // Time graph execution
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 1000; i++) {
        cudaGraphLaunch(graphExec, stream);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    printf("Graph with 2 kernels\n");
    printf("Launches: 1000\n");
    printf("Total time: %.2f ms\n", ms);
    printf("Time per launch: %.3f ms\n", ms / 1000.0);

    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
