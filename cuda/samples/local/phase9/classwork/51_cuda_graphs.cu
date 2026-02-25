
/*
 * CUDA Graphs - Capture and replay kernel sequences
 *
 * CUDA Graphs allow you to define a workflow once and replay it multiple
 * times with reduced overhead. This is particularly useful for repeated
 * kernel launches with the same structure.
 *
 * Benefits:
 * - Reduced CPU overhead
 * - Better optimization opportunities
 * - Simplified code for repeated operations
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Kernel 1: Initialize array
__global__ void initKernel(float *data, int n, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = value;
    }
}

// Kernel 2: Scale array by factor
__global__ void scaleKernel(float *data, int n, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= scale;
    }
}

// Kernel 3: Add offset to array
__global__ void addKernel(float *data, int n, float offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += offset;
    }
}

// Kernel 4: Compute square
__global__ void squareKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * data[idx];
    }
}

int main() {
    printf("=== CUDA Graphs Demo ===\n\n");

    // Problem size
    const int N = 1024 * 1024;  // 1M elements
    const size_t bytes = N * sizeof(float);

    // Allocate device memory
    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));

    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Create events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("Array size: %d elements\n", N);
    printf("Grid: %d blocks, Block: %d threads\n\n", blocksPerGrid, threadsPerBlock);

    // --- Test 1: Traditional kernel launches ---
    printf("Test 1: Traditional Kernel Launches\n");
    printf("Running 100 iterations...\n");

    CUDA_CHECK(cudaEventRecord(start, stream));

    for (int i = 0; i < 100; i++) {
        initKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_data, N, 1.0f);
        scaleKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_data, N, 2.0f);
        addKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_data, N, 3.0f);
        squareKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_data, N);
    }

    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float traditionalTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&traditionalTime, start, stop));
    printf("Total time: %.3f ms\n", traditionalTime);
    printf("Average per iteration: %.3f ms\n\n", traditionalTime / 100);

    // --- Test 2: CUDA Graph - Stream Capture ---
    printf("Test 2: CUDA Graph (Stream Capture)\n");

    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    // Capture the sequence of operations
    printf("Capturing graph...\n");
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    initKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_data, N, 1.0f);
    scaleKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_data, N, 2.0f);
    addKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_data, N, 3.0f);
    squareKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_data, N);

    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

    // Instantiate the graph
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    printf("Running 100 iterations...\n");

    CUDA_CHECK(cudaEventRecord(start, stream));

    for (int i = 0; i < 100; i++) {
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
    }

    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float graphTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&graphTime, start, stop));
    printf("Total time: %.3f ms\n", graphTime);
    printf("Average per iteration: %.3f ms\n\n", graphTime / 100);

    // --- Test 3: CUDA Graph - Manual Construction ---
    printf("Test 3: CUDA Graph (Manual Construction)\n");

    cudaGraph_t manualGraph;
    cudaGraphExec_t manualGraphExec;

    CUDA_CHECK(cudaGraphCreate(&manualGraph, 0));

    // Create nodes manually
    cudaGraphNode_t initNode, scaleNode, addNode, squareNode;
    cudaKernelNodeParams initParams = {0};
    cudaKernelNodeParams scaleParams = {0};
    cudaKernelNodeParams addParams = {0};
    cudaKernelNodeParams squareParams = {0};

    // Setup init kernel parameters
    void *initArgs[] = {&d_data, &N, &(float){1.0f}};
    initParams.func = (void*)initKernel;
    initParams.gridDim = dim3(blocksPerGrid);
    initParams.blockDim = dim3(threadsPerBlock);
    initParams.kernelParams = initArgs;

    // Setup scale kernel parameters
    void *scaleArgs[] = {&d_data, &N, &(float){2.0f}};
    scaleParams.func = (void*)scaleKernel;
    scaleParams.gridDim = dim3(blocksPerGrid);
    scaleParams.blockDim = dim3(threadsPerBlock);
    scaleParams.kernelParams = scaleArgs;

    // Setup add kernel parameters
    void *addArgs[] = {&d_data, &N, &(float){3.0f}};
    addParams.func = (void*)addKernel;
    addParams.gridDim = dim3(blocksPerGrid);
    addParams.blockDim = dim3(threadsPerBlock);
    addParams.kernelParams = addArgs;

    // Setup square kernel parameters
    void *squareArgs[] = {&d_data, &N};
    squareParams.func = (void*)squareKernel;
    squareParams.gridDim = dim3(blocksPerGrid);
    squareParams.blockDim = dim3(threadsPerBlock);
    squareParams.kernelParams = squareArgs;

    // Add nodes to graph with dependencies
    CUDA_CHECK(cudaGraphAddKernelNode(&initNode, manualGraph, NULL, 0, &initParams));
    CUDA_CHECK(cudaGraphAddKernelNode(&scaleNode, manualGraph, &initNode, 1, &scaleParams));
    CUDA_CHECK(cudaGraphAddKernelNode(&addNode, manualGraph, &scaleNode, 1, &addParams));
    CUDA_CHECK(cudaGraphAddKernelNode(&squareNode, manualGraph, &addNode, 1, &squareParams));

    // Instantiate the manually constructed graph
    CUDA_CHECK(cudaGraphInstantiate(&manualGraphExec, manualGraph, NULL, NULL, 0));

    printf("Running 100 iterations...\n");

    CUDA_CHECK(cudaEventRecord(start, stream));

    for (int i = 0; i < 100; i++) {
        CUDA_CHECK(cudaGraphLaunch(manualGraphExec, stream));
    }

    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float manualGraphTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&manualGraphTime, start, stop));
    printf("Total time: %.3f ms\n", manualGraphTime);
    printf("Average per iteration: %.3f ms\n\n", manualGraphTime / 100);

    // Performance comparison
    printf("=== Performance Summary ===\n");
    printf("Traditional launches: %.3f ms/iter\n", traditionalTime / 100);
    printf("Stream capture graph: %.3f ms/iter (%.1fx speedup)\n",
           graphTime / 100, traditionalTime / graphTime);
    printf("Manual graph:         %.3f ms/iter (%.1fx speedup)\n\n",
           manualGraphTime / 100, traditionalTime / manualGraphTime);

    // Verify final result
    float *h_result = (float*)malloc(bytes);
    CUDA_CHECK(cudaMemcpy(h_result, d_data, bytes, cudaMemcpyDeviceToHost));

    // Expected: ((1.0 * 2.0) + 3.0)^2 = 5.0^2 = 25.0
    float expected = 25.0f;
    bool correct = true;
    for (int i = 0; i < 10; i++) {
        if (fabsf(h_result[i] - expected) > 1e-5) {
            printf("Verification failed at %d: got %f, expected %f\n", i, h_result[i], expected);
            correct = false;
            break;
        }
    }

    if (correct) {
        printf("Verification: PASSED\n");
    }

    // Cleanup
    CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaGraphExecDestroy(manualGraphExec));
    CUDA_CHECK(cudaGraphDestroy(manualGraph));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_data));
    free(h_result);

    printf("\nNote: CUDA Graphs are most beneficial when:\n");
    printf("  - Same sequence of operations repeated many times\n");
    printf("  - CPU launch overhead is significant\n");
    printf("  - Operations are small and numerous\n");

    return 0;
}
