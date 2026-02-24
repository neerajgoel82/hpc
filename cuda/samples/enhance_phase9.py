#!/usr/bin/env python3
"""
Generate real implementations for Phase 9 modern CUDA features.
Creates complete, working CUDA samples with proper error handling and timing.
"""

import os

# Target directory
OUTPUT_DIR = "local/phase9"

# CUDA samples with real implementations
CUDA_SAMPLES = {
    "50_dynamic_parallelism.cu": """
/*
 * Dynamic Parallelism - Parent kernel launches child kernels
 *
 * This demonstrates CUDA Dynamic Parallelism where GPU kernels can launch
 * other kernels directly without CPU involvement. This is useful for
 * recursive algorithms, adaptive mesh refinement, and dynamic workloads.
 *
 * Compile with: nvcc -arch=sm_35 -rdc=true 50_dynamic_parallelism.cu -o dynamic_parallelism
 * Note: Requires compute capability 3.5 or higher
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \\
    do { \\
        cudaError_t err = call; \\
        if (err != cudaSuccess) { \\
            fprintf(stderr, "CUDA error at %s:%d: %s\\n", __FILE__, __LINE__, \\
                    cudaGetErrorString(err)); \\
            exit(EXIT_FAILURE); \\
        } \\
    } while(0)

// Child kernel - processes a sub-range of data
__global__ void childKernel(int *data, int offset, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        // Simple computation: square the index and add offset
        data[offset + idx] = (idx * idx) + offset;
    }
}

// Parent kernel - launches multiple child kernels
__global__ void parentKernel(int *data, int n, int childSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int numChildren = (n + childSize - 1) / childSize;

    if (idx < numChildren) {
        int offset = idx * childSize;
        int size = min(childSize, n - offset);

        // Calculate grid and block dimensions for child kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

        // Launch child kernel from GPU
        childKernel<<<blocksPerGrid, threadsPerBlock>>>(data, offset, size);

        // Child kernel launches are asynchronous, but we can sync if needed
        // cudaDeviceSynchronize() would wait for child to complete
    }
}

// Verification kernel - simple operation for comparison
__global__ void simpleKernel(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // Same computation as child kernel
        data[idx] = (idx * idx);
    }
}

int main() {
    printf("=== CUDA Dynamic Parallelism Demo ===\\n\\n");

    // Check if device supports dynamic parallelism
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("Device: %s\\n", prop.name);
    printf("Compute Capability: %d.%d\\n", prop.major, prop.minor);

    if (prop.major < 3 || (prop.major == 3 && prop.minor < 5)) {
        printf("Error: Dynamic Parallelism requires compute capability 3.5 or higher\\n");
        return 1;
    }

    // Problem size
    const int N = 1024 * 1024;  // 1M elements
    const int childSize = 256;   // Each child processes 256 elements
    const size_t bytes = N * sizeof(int);

    printf("Array size: %d elements\\n", N);
    printf("Child kernel size: %d elements\\n\\n", childSize);

    // Allocate host memory
    int *h_data = (int*)malloc(bytes);
    int *h_verify = (int*)malloc(bytes);

    // Allocate device memory
    int *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));

    // Initialize data
    for (int i = 0; i < N; i++) {
        h_data[i] = 0;
    }

    // Create events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // --- Test 1: Dynamic Parallelism ---
    printf("Test 1: Dynamic Parallelism\\n");
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    int numParents = (N + childSize - 1) / childSize;
    int threadsPerBlock = 256;
    int blocksPerGrid = (numParents + threadsPerBlock - 1) / threadsPerBlock;

    printf("Launching %d parent threads in %d blocks\\n", numParents, blocksPerGrid);

    CUDA_CHECK(cudaEventRecord(start));
    parentKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, N, childSize);
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Time: %.3f ms\\n", ms);

    CUDA_CHECK(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));

    // --- Test 2: Simple kernel for comparison ---
    printf("\\nTest 2: Simple Kernel (no dynamic parallelism)\\n");
    CUDA_CHECK(cudaMemset(d_data, 0, bytes));

    threadsPerBlock = 256;
    blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("Launching %d threads in %d blocks\\n", N, blocksPerGrid);

    CUDA_CHECK(cudaEventRecord(start));
    simpleKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Time: %.3f ms\\n", ms);

    CUDA_CHECK(cudaMemcpy(h_verify, d_data, bytes, cudaMemcpyDeviceToHost));

    // Verify results
    printf("\\nVerifying results...\\n");
    int errors = 0;
    for (int i = 0; i < N && errors < 10; i++) {
        int expected = i * i;
        if (h_data[i] != expected) {
            printf("Mismatch at %d: got %d, expected %d\\n", i, h_data[i], expected);
            errors++;
        }
    }

    if (errors == 0) {
        printf("SUCCESS: All results match!\\n");
    } else {
        printf("ERRORS: Found %d mismatches\\n", errors);
    }

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));
    free(h_data);
    free(h_verify);

    printf("\\nNote: Dynamic parallelism adds overhead. It's beneficial when:\\n");
    printf("  - Workload is highly irregular or adaptive\\n");
    printf("  - Cost of CPU-GPU synchronization is high\\n");
    printf("  - Algorithm is naturally recursive\\n");

    return 0;
}
""",

    "51_cuda_graphs.cu": """
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

#define CUDA_CHECK(call) \\
    do { \\
        cudaError_t err = call; \\
        if (err != cudaSuccess) { \\
            fprintf(stderr, "CUDA error at %s:%d: %s\\n", __FILE__, __LINE__, \\
                    cudaGetErrorString(err)); \\
            exit(EXIT_FAILURE); \\
        } \\
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
    printf("=== CUDA Graphs Demo ===\\n\\n");

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

    printf("Array size: %d elements\\n", N);
    printf("Grid: %d blocks, Block: %d threads\\n\\n", blocksPerGrid, threadsPerBlock);

    // --- Test 1: Traditional kernel launches ---
    printf("Test 1: Traditional Kernel Launches\\n");
    printf("Running 100 iterations...\\n");

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
    printf("Total time: %.3f ms\\n", traditionalTime);
    printf("Average per iteration: %.3f ms\\n\\n", traditionalTime / 100);

    // --- Test 2: CUDA Graph - Stream Capture ---
    printf("Test 2: CUDA Graph (Stream Capture)\\n");

    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    // Capture the sequence of operations
    printf("Capturing graph...\\n");
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    initKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_data, N, 1.0f);
    scaleKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_data, N, 2.0f);
    addKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_data, N, 3.0f);
    squareKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_data, N);

    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

    // Instantiate the graph
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    printf("Running 100 iterations...\\n");

    CUDA_CHECK(cudaEventRecord(start, stream));

    for (int i = 0; i < 100; i++) {
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
    }

    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float graphTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&graphTime, start, stop));
    printf("Total time: %.3f ms\\n", graphTime);
    printf("Average per iteration: %.3f ms\\n\\n", graphTime / 100);

    // --- Test 3: CUDA Graph - Manual Construction ---
    printf("Test 3: CUDA Graph (Manual Construction)\\n");

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

    printf("Running 100 iterations...\\n");

    CUDA_CHECK(cudaEventRecord(start, stream));

    for (int i = 0; i < 100; i++) {
        CUDA_CHECK(cudaGraphLaunch(manualGraphExec, stream));
    }

    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float manualGraphTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&manualGraphTime, start, stop));
    printf("Total time: %.3f ms\\n", manualGraphTime);
    printf("Average per iteration: %.3f ms\\n\\n", manualGraphTime / 100);

    // Performance comparison
    printf("=== Performance Summary ===\\n");
    printf("Traditional launches: %.3f ms/iter\\n", traditionalTime / 100);
    printf("Stream capture graph: %.3f ms/iter (%.1fx speedup)\\n",
           graphTime / 100, traditionalTime / graphTime);
    printf("Manual graph:         %.3f ms/iter (%.1fx speedup)\\n\\n",
           manualGraphTime / 100, traditionalTime / manualGraphTime);

    // Verify final result
    float *h_result = (float*)malloc(bytes);
    CUDA_CHECK(cudaMemcpy(h_result, d_data, bytes, cudaMemcpyDeviceToHost));

    // Expected: ((1.0 * 2.0) + 3.0)^2 = 5.0^2 = 25.0
    float expected = 25.0f;
    bool correct = true;
    for (int i = 0; i < 10; i++) {
        if (fabsf(h_result[i] - expected) > 1e-5) {
            printf("Verification failed at %d: got %f, expected %f\\n", i, h_result[i], expected);
            correct = false;
            break;
        }
    }

    if (correct) {
        printf("Verification: PASSED\\n");
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

    printf("\\nNote: CUDA Graphs are most beneficial when:\\n");
    printf("  - Same sequence of operations repeated many times\\n");
    printf("  - CPU launch overhead is significant\\n");
    printf("  - Operations are small and numerous\\n");

    return 0;
}
""",

    "52_mps_demo.cu": """
/*
 * Multi-Process Service (MPS) Demo
 *
 * MPS allows multiple processes to share GPU resources efficiently.
 * This sample demonstrates the concept and provides information about MPS.
 *
 * To actually use MPS, you need to:
 * 1. Start MPS daemon: nvidia-cuda-mps-control -d
 * 2. Run multiple processes
 * 3. Stop MPS: echo quit | nvidia-cuda-mps-control
 *
 * Without MPS: Multiple processes time-slice the GPU (poor utilization)
 * With MPS: Multiple processes share GPU simultaneously (better utilization)
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <unistd.h>

#define CUDA_CHECK(call) \\
    do { \\
        cudaError_t err = call; \\
        if (err != cudaSuccess) { \\
            fprintf(stderr, "CUDA error at %s:%d: %s\\n", __FILE__, __LINE__, \\
                    cudaGetErrorString(err)); \\
            exit(EXIT_FAILURE); \\
        } \\
    } while(0)

// Kernel that does some work (matrix-vector multiply)
__global__ void matVecMul(float *A, float *x, float *y, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n) {
        float sum = 0.0f;
        for (int col = 0; col < n; col++) {
            sum += A[row * n + col] * x[col];
        }
        y[row] = sum;
    }
}

// Kernel that keeps GPU busy
__global__ void busyKernel(float *data, int n, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float value = data[idx];

        // Perform many computations to keep GPU busy
        for (int i = 0; i < iterations; i++) {
            value = sinf(value) * cosf(value) + 1.0f;
            value = sqrtf(fabsf(value));
        }

        data[idx] = value;
    }
}

void checkMPSStatus() {
    printf("=== MPS Status Check ===\\n\\n");

    // Check if MPS is available
    FILE *fp = popen("nvidia-smi -q -d COMPUTE", "r");
    if (fp != NULL) {
        char line[256];
        bool foundMPS = false;

        while (fgets(line, sizeof(line), fp) != NULL) {
            if (strstr(line, "MPS") != NULL) {
                printf("%s", line);
                foundMPS = true;
            }
        }

        pclose(fp);

        if (!foundMPS) {
            printf("MPS status: Not detected or not running\\n");
        }
    } else {
        printf("Could not check MPS status (nvidia-smi not available)\\n");
    }

    printf("\\n");
}

void printMPSInfo() {
    printf("=== About CUDA Multi-Process Service (MPS) ===\\n\\n");

    printf("What is MPS?\\n");
    printf("  MPS is a client-server runtime that allows multiple processes\\n");
    printf("  to share a single GPU context. This enables better GPU utilization\\n");
    printf("  when multiple small workloads run concurrently.\\n\\n");

    printf("Benefits:\\n");
    printf("  - Reduced context switching overhead\\n");
    printf("  - Better GPU utilization for small kernels\\n");
    printf("  - Multiple processes can use GPU simultaneously\\n");
    printf("  - Lower latency for concurrent workloads\\n\\n");

    printf("When to use MPS:\\n");
    printf("  - Multiple MPI ranks per node\\n");
    printf("  - Multiple small applications sharing GPU\\n");
    printf("  - HPC workloads with many processes\\n");
    printf("  - Microservices architecture\\n\\n");

    printf("How to enable MPS:\\n");
    printf("  1. Export CUDA_VISIBLE_DEVICES=0 (or your GPU ID)\\n");
    printf("  2. Start MPS daemon: nvidia-cuda-mps-control -d\\n");
    printf("  3. Run your applications\\n");
    printf("  4. Stop MPS: echo quit | nvidia-cuda-mps-control\\n\\n");

    printf("Limitations:\\n");
    printf("  - Requires Volta or newer for full features\\n");
    printf("  - Limited debugger support\\n");
    printf("  - All processes must be from same user\\n");
    printf("  - Some features require privileged access\\n\\n");
}

int main(int argc, char **argv) {
    printf("=== CUDA MPS Demo ===\\n\\n");

    // Get process ID for identification
    int pid = getpid();
    int processNum = 0;

    if (argc > 1) {
        processNum = atoi(argv[1]);
    }

    printf("Process ID: %d\\n", pid);
    printf("Process Number: %d\\n\\n", processNum);

    // Check device properties
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("Device: %s\\n", prop.name);
    printf("Compute Capability: %d.%d\\n", prop.major, prop.minor);
    printf("Multi-Process Service: %s\\n\\n",
           prop.major >= 7 ? "Fully Supported (Volta+)" :
           prop.major >= 3 ? "Partially Supported" : "Not Supported");

    // Problem size
    const int N = 2048;  // Matrix size
    const size_t matrixBytes = N * N * sizeof(float);
    const size_t vectorBytes = N * sizeof(float);

    // Allocate device memory
    float *d_A, *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_A, matrixBytes));
    CUDA_CHECK(cudaMalloc(&d_x, vectorBytes));
    CUDA_CHECK(cudaMalloc(&d_y, vectorBytes));

    // Initialize data
    CUDA_CHECK(cudaMemset(d_A, 0, matrixBytes));
    CUDA_CHECK(cudaMemset(d_x, 0, vectorBytes));

    // Create events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("Running workload for process %d...\\n", processNum);
    printf("Matrix size: %d x %d\\n", N, N);
    printf("Grid: %d blocks, Block: %d threads\\n\\n", blocksPerGrid, threadsPerBlock);

    // Run multiple iterations
    const int iterations = 50;

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < iterations; i++) {
        matVecMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_x, d_y, N);

        // Add some busy work
        busyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_y, N, 100);

        // Small delay to simulate real application
        if (i % 10 == 0) {
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    printf("Completed %d iterations\\n", iterations);
    printf("Total time: %.3f ms\\n", ms);
    printf("Average time per iteration: %.3f ms\\n\\n", ms / iterations);

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    // Print MPS information
    if (processNum == 0) {
        checkMPSStatus();
        printMPSInfo();

        printf("=== To Test MPS ===\\n\\n");
        printf("Run without MPS:\\n");
        printf("  Terminal 1: ./mps_demo 0\\n");
        printf("  Terminal 2: ./mps_demo 1\\n");
        printf("  (Processes will time-slice the GPU)\\n\\n");

        printf("Run with MPS:\\n");
        printf("  1. nvidia-cuda-mps-control -d\\n");
        printf("  2. Terminal 1: ./mps_demo 0\\n");
        printf("  3. Terminal 2: ./mps_demo 1\\n");
        printf("  4. echo quit | nvidia-cuda-mps-control\\n");
        printf("  (Processes will share GPU simultaneously)\\n\\n");

        printf("Compare execution times - MPS should show better total throughput\\n");
        printf("when running multiple processes concurrently.\\n");
    }

    return 0;
}
""",

    "53_mixed_precision.cu": """
/*
 * Mixed Precision Computing - FP16 and FP32 operations
 *
 * Mixed precision uses both FP16 (half precision) and FP32 (single precision)
 * to optimize both performance and accuracy. FP16 offers:
 * - 2x memory bandwidth
 * - 2x throughput on Tensor Cores
 * - Lower power consumption
 *
 * But requires careful handling to maintain accuracy.
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CUDA_CHECK(call) \\
    do { \\
        cudaError_t err = call; \\
        if (err != cudaSuccess) { \\
            fprintf(stderr, "CUDA error at %s:%d: %s\\n", __FILE__, __LINE__, \\
                    cudaGetErrorString(err)); \\
            exit(EXIT_FAILURE); \\
        } \\
    } while(0)

// Vector addition in FP32 (baseline)
__global__ void vecAddFP32(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Vector addition in FP16
__global__ void vecAddFP16(__half *a, __half *b, __half *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = __hadd(a[idx], b[idx]);
    }
}

// Mixed precision: FP16 input, FP32 accumulation, FP16 output
__global__ void vecAddMixed(__half *a, __half *b, __half *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Convert to FP32 for addition
        float a_fp32 = __half2float(a[idx]);
        float b_fp32 = __half2float(b[idx]);
        float sum = a_fp32 + b_fp32;
        // Convert back to FP16
        c[idx] = __float2half(sum);
    }
}

// Vector dot product in FP32
__global__ void dotProductFP32(float *a, float *b, float *result, int n) {
    __shared__ float shared[256];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load and multiply
    float temp = 0.0f;
    if (idx < n) {
        temp = a[idx] * b[idx];
    }
    shared[tid] = temp;
    __syncthreads();

    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    // Write result
    if (tid == 0) {
        atomicAdd(result, shared[0]);
    }
}

// Vector dot product - Mixed precision (FP16 input, FP32 accumulation)
__global__ void dotProductMixed(__half *a, __half *b, float *result, int n) {
    __shared__ float shared[256];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load, convert to FP32, and multiply
    float temp = 0.0f;
    if (idx < n) {
        float a_fp32 = __half2float(a[idx]);
        float b_fp32 = __half2float(b[idx]);
        temp = a_fp32 * b_fp32;
    }
    shared[tid] = temp;
    __syncthreads();

    // Reduction in FP32
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    // Write result
    if (tid == 0) {
        atomicAdd(result, shared[0]);
    }
}

// Matrix multiplication in FP32
__global__ void matMulFP32(float *A, float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Matrix multiplication - Mixed precision
__global__ void matMulMixed(__half *A, __half *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            float a_val = __half2float(A[row * n + k]);
            float b_val = __half2float(B[k * n + col]);
            sum += a_val * b_val;
        }
        C[row * n + col] = sum;
    }
}

int main() {
    printf("=== Mixed Precision Computing Demo ===\\n\\n");

    // Check device capabilities
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("Device: %s\\n", prop.name);
    printf("Compute Capability: %d.%d\\n", prop.major, prop.minor);

    // Check FP16 support
    bool hasFP16 = (prop.major >= 6);
    if (!hasFP16) {
        printf("Warning: FP16 operations may be slow on this device (requires Pascal or newer)\\n");
    }
    printf("\\n");

    // Problem sizes
    const int N = 1024 * 1024;  // 1M elements for vectors
    const int M = 1024;          // 1K x 1K matrix

    // Create events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // --- Test 1: Vector Addition ---
    printf("=== Test 1: Vector Addition ===\\n");
    printf("Vector size: %d elements\\n\\n", N);

    size_t vecBytesFP32 = N * sizeof(float);
    size_t vecBytesFP16 = N * sizeof(__half);

    // Allocate FP32 arrays
    float *d_a_fp32, *d_b_fp32, *d_c_fp32;
    CUDA_CHECK(cudaMalloc(&d_a_fp32, vecBytesFP32));
    CUDA_CHECK(cudaMalloc(&d_b_fp32, vecBytesFP32));
    CUDA_CHECK(cudaMalloc(&d_c_fp32, vecBytesFP32));

    // Allocate FP16 arrays
    __half *d_a_fp16, *d_b_fp16, *d_c_fp16;
    CUDA_CHECK(cudaMalloc(&d_a_fp16, vecBytesFP16));
    CUDA_CHECK(cudaMalloc(&d_b_fp16, vecBytesFP16));
    CUDA_CHECK(cudaMalloc(&d_c_fp16, vecBytesFP16));

    // Initialize with some values
    CUDA_CHECK(cudaMemset(d_a_fp32, 0, vecBytesFP32));
    CUDA_CHECK(cudaMemset(d_b_fp32, 0, vecBytesFP32));
    CUDA_CHECK(cudaMemset(d_a_fp16, 0, vecBytesFP16));
    CUDA_CHECK(cudaMemset(d_b_fp16, 0, vecBytesFP16));

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // FP32 addition
    printf("FP32 Vector Addition:\\n");
    CUDA_CHECK(cudaEventRecord(start));
    vecAddFP32<<<blocksPerGrid, threadsPerBlock>>>(d_a_fp32, d_b_fp32, d_c_fp32, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    float fp32Time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&fp32Time, start, stop));
    printf("  Time: %.3f ms\\n", fp32Time);
    printf("  Bandwidth: %.2f GB/s\\n", 3.0 * vecBytesFP32 / fp32Time / 1e6);

    // FP16 addition
    printf("\\nFP16 Vector Addition:\\n");
    CUDA_CHECK(cudaEventRecord(start));
    vecAddFP16<<<blocksPerGrid, threadsPerBlock>>>(d_a_fp16, d_b_fp16, d_c_fp16, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    float fp16Time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&fp16Time, start, stop));
    printf("  Time: %.3f ms\\n", fp16Time);
    printf("  Bandwidth: %.2f GB/s\\n", 3.0 * vecBytesFP16 / fp16Time / 1e6);
    printf("  Speedup: %.2fx\\n", fp32Time / fp16Time);
    printf("  Memory saved: %.2f MB (%.1f%%)\\n",
           (vecBytesFP32 - vecBytesFP16) / 1e6 * 3, 50.0);

    // Mixed precision addition
    printf("\\nMixed Precision Vector Addition:\\n");
    CUDA_CHECK(cudaEventRecord(start));
    vecAddMixed<<<blocksPerGrid, threadsPerBlock>>>(d_a_fp16, d_b_fp16, d_c_fp16, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    float mixedTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&mixedTime, start, stop));
    printf("  Time: %.3f ms\\n", mixedTime);
    printf("  Speedup: %.2fx\\n\\n", fp32Time / mixedTime);

    // --- Test 2: Dot Product ---
    printf("=== Test 2: Dot Product ===\\n");

    float *d_result_fp32;
    CUDA_CHECK(cudaMalloc(&d_result_fp32, sizeof(float)));

    // FP32 dot product
    printf("FP32 Dot Product:\\n");
    CUDA_CHECK(cudaMemset(d_result_fp32, 0, sizeof(float)));
    CUDA_CHECK(cudaEventRecord(start));
    dotProductFP32<<<blocksPerGrid, threadsPerBlock>>>(d_a_fp32, d_b_fp32, d_result_fp32, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventElapsedTime(&fp32Time, start, stop));
    printf("  Time: %.3f ms\\n", fp32Time);

    // Mixed precision dot product
    printf("\\nMixed Precision Dot Product:\\n");
    CUDA_CHECK(cudaMemset(d_result_fp32, 0, sizeof(float)));
    CUDA_CHECK(cudaEventRecord(start));
    dotProductMixed<<<blocksPerGrid, threadsPerBlock>>>(d_a_fp16, d_b_fp16, d_result_fp32, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventElapsedTime(&mixedTime, start, stop));
    printf("  Time: %.3f ms\\n", mixedTime);
    printf("  Speedup: %.2fx\\n\\n", fp32Time / mixedTime);

    // --- Test 3: Matrix Multiplication ---
    printf("=== Test 3: Matrix Multiplication ===\\n");
    printf("Matrix size: %d x %d\\n\\n", M, M);

    size_t matBytesFP32 = M * M * sizeof(float);
    size_t matBytesFP16 = M * M * sizeof(__half);

    // Allocate matrices
    float *d_A_fp32, *d_B_fp32, *d_C_fp32;
    CUDA_CHECK(cudaMalloc(&d_A_fp32, matBytesFP32));
    CUDA_CHECK(cudaMalloc(&d_B_fp32, matBytesFP32));
    CUDA_CHECK(cudaMalloc(&d_C_fp32, matBytesFP32));

    __half *d_A_fp16, *d_B_fp16;
    CUDA_CHECK(cudaMalloc(&d_A_fp16, matBytesFP16));
    CUDA_CHECK(cudaMalloc(&d_B_fp16, matBytesFP16));

    // Initialize
    CUDA_CHECK(cudaMemset(d_A_fp32, 0, matBytesFP32));
    CUDA_CHECK(cudaMemset(d_B_fp32, 0, matBytesFP32));
    CUDA_CHECK(cudaMemset(d_A_fp16, 0, matBytesFP16));
    CUDA_CHECK(cudaMemset(d_B_fp16, 0, matBytesFP16));

    dim3 blockDim(16, 16);
    dim3 gridDim((M + 15) / 16, (M + 15) / 16);

    // FP32 matrix multiply
    printf("FP32 Matrix Multiply:\\n");
    CUDA_CHECK(cudaEventRecord(start));
    matMulFP32<<<gridDim, blockDim>>>(d_A_fp32, d_B_fp32, d_C_fp32, M);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventElapsedTime(&fp32Time, start, stop));
    float fp32Gflops = 2.0 * M * M * M / fp32Time / 1e6;
    printf("  Time: %.3f ms\\n", fp32Time);
    printf("  Performance: %.2f GFLOPS\\n", fp32Gflops);

    // Mixed precision matrix multiply
    printf("\\nMixed Precision Matrix Multiply:\\n");
    CUDA_CHECK(cudaEventRecord(start));
    matMulMixed<<<gridDim, blockDim>>>(d_A_fp16, d_B_fp16, d_C_fp32, M);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventElapsedTime(&mixedTime, start, stop));
    float mixedGflops = 2.0 * M * M * M / mixedTime / 1e6;
    printf("  Time: %.3f ms\\n", mixedTime);
    printf("  Performance: %.2f GFLOPS\\n", mixedGflops);
    printf("  Speedup: %.2fx\\n", fp32Time / mixedTime);
    printf("  Input memory saved: %.2f MB (%.1f%%)\\n\\n",
           (matBytesFP32 - matBytesFP16) / 1e6 * 2, 50.0);

    // Summary
    printf("=== Summary ===\\n");
    printf("Mixed precision benefits:\\n");
    printf("  - 2x memory bandwidth (stores half the data)\\n");
    printf("  - Faster computation (especially on Tensor Cores)\\n");
    printf("  - Lower power consumption\\n\\n");

    printf("Best practices:\\n");
    printf("  - Use FP16 for storage and bandwidth-bound operations\\n");
    printf("  - Use FP32 for accumulation to maintain accuracy\\n");
    printf("  - Use loss scaling in deep learning to prevent underflow\\n");
    printf("  - Profile to find the right balance for your application\\n");

    // Cleanup
    CUDA_CHECK(cudaFree(d_a_fp32));
    CUDA_CHECK(cudaFree(d_b_fp32));
    CUDA_CHECK(cudaFree(d_c_fp32));
    CUDA_CHECK(cudaFree(d_a_fp16));
    CUDA_CHECK(cudaFree(d_b_fp16));
    CUDA_CHECK(cudaFree(d_c_fp16));
    CUDA_CHECK(cudaFree(d_result_fp32));
    CUDA_CHECK(cudaFree(d_A_fp32));
    CUDA_CHECK(cudaFree(d_B_fp32));
    CUDA_CHECK(cudaFree(d_C_fp32));
    CUDA_CHECK(cudaFree(d_A_fp16));
    CUDA_CHECK(cudaFree(d_B_fp16));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
""",

    "54_tensor_cores.cu": """
/*
 * Tensor Cores Demo - Hardware-accelerated matrix operations
 *
 * Tensor Cores are specialized hardware units for matrix multiply-accumulate
 * operations. They provide massive speedup for deep learning workloads.
 *
 * Available on:
 * - Volta (V100): FP16 input, FP32 accumulation
 * - Turing (RTX 20xx): FP16, INT8, INT4
 * - Ampere (A100, RTX 30xx): FP16, BF16, TF32, INT8, INT4, INT1
 * - Hopper (H100): FP8, FP16, BF16, TF32, INT8
 *
 * This sample demonstrates the WMMA (Warp Matrix Multiply-Accumulate) API.
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Check if WMMA is available (requires sm_70 or higher)
#if __CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__)
#include <mma.h>
#define WMMA_AVAILABLE 1
#else
#define WMMA_AVAILABLE 0
#endif

#define CUDA_CHECK(call) \\
    do { \\
        cudaError_t err = call; \\
        if (err != cudaSuccess) { \\
            fprintf(stderr, "CUDA error at %s:%d: %s\\n", __FILE__, __LINE__, \\
                    cudaGetErrorString(err)); \\
            exit(EXIT_FAILURE); \\
        } \\
    } while(0)

// Simple matrix multiply without Tensor Cores (FP32)
__global__ void matmulFP32(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Matrix multiply with FP16 (no Tensor Cores)
__global__ void matmulFP16(__half *A, __half *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            float a = __half2float(A[row * K + i]);
            float b = __half2float(B[i * N + col]);
            sum += a * b;
        }
        C[row * N + col] = sum;
    }
}

#if WMMA_AVAILABLE
using namespace nvcuda::wmma;

// Matrix multiply using WMMA (Tensor Cores)
// WMMA works on 16x16x16 matrix fragments
__global__ void matmulWMMA(__half *A, __half *B, float *C, int M, int N, int K) {
    // Warp and lane identification
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y) / 32;
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x);

    // Declare the fragments
    fragment<matrix_a, 16, 16, 16, __half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, __half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> acc_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;

    // Initialize the output to zero
    fill_fragment(acc_frag, 0.0f);

    // Loop over K dimension in chunks of 16
    for (int k = 0; k < K; k += 16) {
        int aRow = warpM * 16;
        int aCol = k;
        int bRow = k;
        int bCol = warpN * 16;

        // Bounds checking
        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            // Load the inputs
            load_matrix_sync(a_frag, A + aRow * K + aCol, K);
            load_matrix_sync(b_frag, B + bRow * N + bCol, N);

            // Perform the matrix multiply-accumulate
            mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // Store the output
    int cRow = warpM * 16;
    int cCol = warpN * 16;

    if (cRow < M && cCol < N) {
        store_matrix_sync(C + cRow * N + cCol, acc_frag, N, mem_row_major);
    }
}
#endif

void printTensorCoreInfo() {
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("=== Tensor Core Information ===\\n\\n");
    printf("Device: %s\\n", prop.name);
    printf("Compute Capability: %d.%d\\n\\n", prop.major, prop.minor);

    if (prop.major >= 7) {
        printf("Tensor Cores: AVAILABLE\\n\\n");

        printf("Supported operations by architecture:\\n");
        if (prop.major == 7 && prop.minor == 0) {
            printf("  Volta (sm_70):\\n");
            printf("    - FP16 input, FP32 accumulation\\n");
            printf("    - D = A * B + C (16x16x16 tiles)\\n");
        } else if (prop.major == 7 && prop.minor == 5) {
            printf("  Turing (sm_75):\\n");
            printf("    - FP16, INT8, INT4, INT1\\n");
            printf("    - 16x16x16 and 8x8x32 tiles\\n");
        } else if (prop.major == 8 && prop.minor == 0) {
            printf("  Ampere (sm_80):\\n");
            printf("    - FP64, TF32, BF16, FP16, INT8, INT4, INT1\\n");
            printf("    - Multiple tile sizes\\n");
            printf("    - Sparsity support\\n");
        } else if (prop.major == 8 && prop.minor == 6) {
            printf("  Ampere (sm_86) - Gaming:\\n");
            printf("    - TF32, BF16, FP16, INT8, INT4, INT1\\n");
        } else if (prop.major == 9 && prop.minor == 0) {
            printf("  Hopper (sm_90):\\n");
            printf("    - FP8, FP64, TF32, BF16, FP16, INT8\\n");
            printf("    - Thread block clusters\\n");
            printf("    - Tensor memory accelerator\\n");
        }

        printf("\\nPerformance characteristics:\\n");
        printf("  - Up to 8x faster than CUDA cores for FP16\\n");
        printf("  - Up to 16x faster for INT8\\n");
        printf("  - Optimized for matrix sizes that are multiples of 16\\n");
        printf("  - Best with mixed precision (FP16 input, FP32 accumulation)\\n");
    } else {
        printf("Tensor Cores: NOT AVAILABLE\\n");
        printf("  Requires compute capability 7.0 or higher (Volta+)\\n");
    }
    printf("\\n");
}

int main() {
    printf("=== Tensor Cores Demo ===\\n\\n");

    printTensorCoreInfo();

    // Check device capabilities
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

#if !WMMA_AVAILABLE
    printf("Warning: This code was not compiled with Tensor Core support\\n");
    printf("Recompile with: nvcc -arch=sm_70 or higher\\n\\n");
#endif

    if (prop.major < 7) {
        printf("This GPU does not support Tensor Cores.\\n");
        printf("Running comparison with standard CUDA cores only.\\n\\n");
    }

    // Matrix dimensions (must be multiples of 16 for WMMA)
    const int M = 1024;  // Rows of A and C
    const int K = 1024;  // Cols of A, Rows of B
    const int N = 1024;  // Cols of B and C

    printf("Matrix dimensions: M=%d, K=%d, N=%d\\n", M, K, N);
    printf("Total operations: %.2f GFLOP\\n\\n", 2.0 * M * N * K / 1e9);

    // Allocate memory
    size_t bytesA_fp32 = M * K * sizeof(float);
    size_t bytesB_fp32 = K * N * sizeof(float);
    size_t bytesC = M * N * sizeof(float);
    size_t bytesA_fp16 = M * K * sizeof(__half);
    size_t bytesB_fp16 = K * N * sizeof(__half);

    float *d_A_fp32, *d_B_fp32, *d_C_fp32;
    __half *d_A_fp16, *d_B_fp16;
    float *d_C_wmma;

    CUDA_CHECK(cudaMalloc(&d_A_fp32, bytesA_fp32));
    CUDA_CHECK(cudaMalloc(&d_B_fp32, bytesB_fp32));
    CUDA_CHECK(cudaMalloc(&d_C_fp32, bytesC));
    CUDA_CHECK(cudaMalloc(&d_A_fp16, bytesA_fp16));
    CUDA_CHECK(cudaMalloc(&d_B_fp16, bytesB_fp16));
    CUDA_CHECK(cudaMalloc(&d_C_wmma, bytesC));

    // Initialize matrices
    CUDA_CHECK(cudaMemset(d_A_fp32, 0, bytesA_fp32));
    CUDA_CHECK(cudaMemset(d_B_fp32, 0, bytesB_fp32));
    CUDA_CHECK(cudaMemset(d_A_fp16, 0, bytesA_fp16));
    CUDA_CHECK(cudaMemset(d_B_fp16, 0, bytesB_fp16));

    // Create events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // --- Test 1: FP32 Standard CUDA Cores ---
    printf("Test 1: FP32 (Standard CUDA Cores)\\n");

    dim3 blockDim(16, 16);
    dim3 gridDim((N + 15) / 16, (M + 15) / 16);

    CUDA_CHECK(cudaEventRecord(start));
    matmulFP32<<<gridDim, blockDim>>>(d_A_fp32, d_B_fp32, d_C_fp32, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    float fp32Time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&fp32Time, start, stop));
    float fp32Gflops = 2.0 * M * N * K / fp32Time / 1e6;

    printf("  Time: %.3f ms\\n", fp32Time);
    printf("  Performance: %.2f GFLOPS\\n\\n", fp32Gflops);

    // --- Test 2: FP16 without Tensor Cores ---
    printf("Test 2: FP16 (CUDA Cores, no Tensor Cores)\\n");

    CUDA_CHECK(cudaEventRecord(start));
    matmulFP16<<<gridDim, blockDim>>>(d_A_fp16, d_B_fp16, d_C_wmma, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    float fp16Time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&fp16Time, start, stop));
    float fp16Gflops = 2.0 * M * N * K / fp16Time / 1e6;

    printf("  Time: %.3f ms\\n", fp16Time);
    printf("  Performance: %.2f GFLOPS\\n", fp16Gflops);
    printf("  Speedup vs FP32: %.2fx\\n\\n", fp32Time / fp16Time);

#if WMMA_AVAILABLE
    // --- Test 3: Tensor Cores with WMMA ---
    if (prop.major >= 7) {
        printf("Test 3: FP16 with Tensor Cores (WMMA)\\n");

        // For WMMA, we need different grid dimensions
        // Each warp processes a 16x16 output tile
        dim3 wmmaBlockDim(32, 4);  // 128 threads per block
        dim3 wmmaGridDim((N + 15) / 16, (M + 15) / 16);

        CUDA_CHECK(cudaEventRecord(start));
        matmulWMMA<<<wmmaGridDim, wmmaBlockDim>>>(d_A_fp16, d_B_fp16, d_C_wmma, M, N, K);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaDeviceSynchronize());

        float wmmaTime = 0;
        CUDA_CHECK(cudaEventElapsedTime(&wmmaTime, start, stop));
        float wmmaGflops = 2.0 * M * N * K / wmmaTime / 1e6;

        printf("  Time: %.3f ms\\n", wmmaTime);
        printf("  Performance: %.2f GFLOPS\\n", wmmaGflops);
        printf("  Speedup vs FP32: %.2fx\\n", fp32Time / wmmaTime);
        printf("  Speedup vs FP16: %.2fx\\n\\n", fp16Time / wmmaTime);

        // Summary
        printf("=== Performance Summary ===\\n");
        printf("FP32 (CUDA Cores):    %6.2f GFLOPS (%.2fx)\\n", fp32Gflops, 1.0f);
        printf("FP16 (CUDA Cores):    %6.2f GFLOPS (%.2fx)\\n", fp16Gflops, fp32Gflops / fp16Gflops);
        printf("FP16 (Tensor Cores):  %6.2f GFLOPS (%.2fx)\\n\\n", wmmaGflops, fp32Gflops / wmmaGflops);
    }
#else
    printf("Test 3: Tensor Cores NOT AVAILABLE\\n");
    printf("  Compile with -arch=sm_70 or higher to enable\\n\\n");
#endif

    printf("=== Notes ===\\n");
    printf("Tensor Cores are best for:\\n");
    printf("  - Deep learning training and inference\\n");
    printf("  - Large matrix multiplications\\n");
    printf("  - Mixed precision workloads\\n");
    printf("  - Batch processing\\n\\n");

    printf("Optimization tips:\\n");
    printf("  - Use dimensions that are multiples of 16\\n");
    printf("  - Keep matrices in FP16 for bandwidth\\n");
    printf("  - Accumulate in FP32 for accuracy\\n");
    printf("  - Use cuBLAS or cuDNN when possible (highly optimized)\\n");

    // Cleanup
    CUDA_CHECK(cudaFree(d_A_fp32));
    CUDA_CHECK(cudaFree(d_B_fp32));
    CUDA_CHECK(cudaFree(d_C_fp32));
    CUDA_CHECK(cudaFree(d_A_fp16));
    CUDA_CHECK(cudaFree(d_B_fp16));
    CUDA_CHECK(cudaFree(d_C_wmma));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
""",

    "55_wmma_gemm.cu": """
/*
 * WMMA GEMM - Warp Matrix Multiply-Accumulate for Tensor Cores
 *
 * This implements a complete GEMM (General Matrix Multiply) using the
 * WMMA API to leverage Tensor Cores. This is the building block for
 * deep learning operations.
 *
 * Operation: C = alpha * A * B + beta * C
 *
 * WMMA operates on small matrix fragments (typically 16x16x16):
 * - matrix_a: 16x16 input matrix A
 * - matrix_b: 16x16 input matrix B
 * - accumulator: 16x16 accumulator/output matrix C
 *
 * Compile: nvcc -arch=sm_70 55_wmma_gemm.cu -o wmma_gemm
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#if __CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__)
#include <mma.h>
#define WMMA_AVAILABLE 1
#else
#define WMMA_AVAILABLE 0
#endif

#define CUDA_CHECK(call) \\
    do { \\
        cudaError_t err = call; \\
        if (err != cudaSuccess) { \\
            fprintf(stderr, "CUDA error at %s:%d: %s\\n", __FILE__, __LINE__, \\
                    cudaGetErrorString(err)); \\
            exit(EXIT_FAILURE); \\
        } \\
    } while(0)

// WMMA tile sizes
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

#if WMMA_AVAILABLE
using namespace nvcuda::wmma;

/*
 * WMMA-based GEMM kernel
 *
 * Each warp computes a WMMA_M x WMMA_N tile of the output matrix.
 * The warp loads tiles from A and B, performs matrix multiply-accumulate,
 * and stores the result to C.
 *
 * Grid and block dimensions:
 * - blockDim: (WARP_SIZE, 4) = 128 threads, 4 warps per block
 * - gridDim: enough blocks to cover the output matrix
 */
__global__ void wmmaGemm(
    const __half *A,    // M x K matrix
    const __half *B,    // K x N matrix
    float *C,           // M x N matrix
    int M, int N, int K,
    float alpha, float beta)
{
    // Warp and lane identification
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // Declare the fragments
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, col_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize accumulator to zero
    fill_fragment(acc_frag, 0.0f);

    // Calculate base positions for this warp's output tile
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;

    // Bounds checking
    if (cRow >= M || cCol >= N) return;

    // Loop over K dimension in WMMA_K chunks
    for (int k = 0; k < K; k += WMMA_K) {
        int aRow = cRow;
        int aCol = k;
        int bRow = k;
        int bCol = cCol;

        // Check bounds for A and B
        if (aCol + WMMA_K <= K && bRow + WMMA_K <= K) {
            // Load matrix fragments from A and B
            load_matrix_sync(a_frag, A + aRow * K + aCol, K);
            load_matrix_sync(b_frag, B + bRow * N + bCol, N);

            // Perform matrix multiply-accumulate: acc = a * b + acc
            mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // Load existing C values if beta != 0
    if (beta != 0.0f) {
        load_matrix_sync(c_frag, C + cRow * N + cCol, N, mem_row_major);

        // Scale C by beta and add to accumulator
        for (int i = 0; i < c_frag.num_elements; i++) {
            acc_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }
    } else {
        // Just scale by alpha
        for (int i = 0; i < acc_frag.num_elements; i++) {
            acc_frag.x[i] = alpha * acc_frag.x[i];
        }
    }

    // Store the result
    store_matrix_sync(C + cRow * N + cCol, acc_frag, N, mem_row_major);
}

/*
 * Optimized WMMA GEMM with tiling for better performance
 *
 * Each warp processes multiple WMMA tiles to improve data reuse
 * and reduce memory traffic.
 */
__global__ void wmmaGemmTiled(
    const __half *A,
    const __half *B,
    float *C,
    int M, int N, int K,
    float alpha, float beta)
{
    // Each block processes a larger tile
    const int BLOCK_ROW_TILES = 4;
    const int BLOCK_COL_TILES = 4;

    // Warp identification within block
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;
    int warpRow = warpId / BLOCK_COL_TILES;
    int warpCol = warpId % BLOCK_COL_TILES;

    // Block position in output matrix
    int blockRowOffset = blockIdx.x * WMMA_M * BLOCK_ROW_TILES;
    int blockColOffset = blockIdx.y * WMMA_N * BLOCK_COL_TILES;

    // This warp's position
    int warpRowOffset = blockRowOffset + warpRow * WMMA_M;
    int warpColOffset = blockColOffset + warpCol * WMMA_N;

    // Bounds checking
    if (warpRowOffset >= M || warpColOffset >= N) return;

    // Declare fragments
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, col_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    // Initialize accumulator
    fill_fragment(acc_frag, 0.0f);

    // Main loop over K dimension
    for (int k = 0; k < K; k += WMMA_K) {
        if (k + WMMA_K <= K) {
            // Load tiles
            load_matrix_sync(a_frag, A + warpRowOffset * K + k, K);
            load_matrix_sync(b_frag, B + k * N + warpColOffset, N);

            // Multiply-accumulate
            mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // Apply alpha and beta
    if (beta != 0.0f) {
        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
        load_matrix_sync(c_frag, C + warpRowOffset * N + warpColOffset, N, mem_row_major);

        for (int i = 0; i < acc_frag.num_elements; i++) {
            acc_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }
    } else {
        for (int i = 0; i < acc_frag.num_elements; i++) {
            acc_frag.x[i] = alpha * acc_frag.x[i];
        }
    }

    // Store result
    store_matrix_sync(C + warpRowOffset * N + warpColOffset, acc_frag, N, mem_row_major);
}
#endif

// Reference CPU implementation for verification
void cpuGemm(const float *A, const float *B, float *C,
             int M, int N, int K, float alpha, float beta) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}

int main() {
    printf("=== WMMA GEMM Demo ===\\n\\n");

    // Check device support
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("Device: %s\\n", prop.name);
    printf("Compute Capability: %d.%d\\n\\n", prop.major, prop.minor);

    if (prop.major < 7) {
        printf("Error: Tensor Cores require compute capability 7.0 or higher\\n");
        return 1;
    }

#if !WMMA_AVAILABLE
    printf("Error: Code not compiled with WMMA support\\n");
    printf("Recompile with: nvcc -arch=sm_70 or higher\\n");
    return 1;
#else

    // Matrix dimensions (must be multiples of 16)
    const int M = 512;
    const int N = 512;
    const int K = 512;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    printf("Matrix dimensions: M=%d, N=%d, K=%d\\n", M, N, K);
    printf("Operation: C = %.1f * A * B + %.1f * C\\n", alpha, beta);
    printf("WMMA tile size: %dx%dx%d\\n", WMMA_M, WMMA_N, WMMA_K);
    printf("Total FLOPs: %.2f GFLOP\\n\\n", 2.0 * M * N * K / 1e9);

    // Allocate host memory
    size_t bytesA_fp32 = M * K * sizeof(float);
    size_t bytesB_fp32 = K * N * sizeof(float);
    size_t bytesC = M * N * sizeof(float);
    size_t bytesA_fp16 = M * K * sizeof(__half);
    size_t bytesB_fp16 = K * N * sizeof(__half);

    float *h_A = (float*)malloc(bytesA_fp32);
    float *h_B = (float*)malloc(bytesB_fp32);
    float *h_C = (float*)malloc(bytesC);
    float *h_C_ref = (float*)malloc(bytesC);

    // Initialize with random values
    for (int i = 0; i < M * K; i++) {
        h_A[i] = (float)(rand() % 100) / 100.0f;
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = (float)(rand() % 100) / 100.0f;
    }
    for (int i = 0; i < M * N; i++) {
        h_C[i] = 0.0f;
        h_C_ref[i] = 0.0f;
    }

    // Convert to FP16
    __half *h_A_fp16 = (__half*)malloc(bytesA_fp16);
    __half *h_B_fp16 = (__half*)malloc(bytesB_fp16);

    for (int i = 0; i < M * K; i++) {
        h_A_fp16[i] = __float2half(h_A[i]);
    }
    for (int i = 0; i < K * N; i++) {
        h_B_fp16[i] = __float2half(h_B[i]);
    }

    // Allocate device memory
    __half *d_A, *d_B;
    float *d_C;

    CUDA_CHECK(cudaMalloc(&d_A, bytesA_fp16));
    CUDA_CHECK(cudaMalloc(&d_B, bytesB_fp16));
    CUDA_CHECK(cudaMalloc(&d_C, bytesC));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A_fp16, bytesA_fp16, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B_fp16, bytesB_fp16, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, bytesC, cudaMemcpyHostToDevice));

    // Create events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // --- Test 1: Basic WMMA GEMM ---
    printf("Test 1: Basic WMMA GEMM\\n");

    dim3 blockDim(32, 4);  // 128 threads per block (4 warps)
    dim3 gridDim((M + WMMA_M - 1) / WMMA_M, (N + WMMA_N - 1) / WMMA_N);

    printf("Grid: (%d, %d), Block: (%d, %d)\\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);

    // Warm-up
    wmmaGemm<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed run
    CUDA_CHECK(cudaEventRecord(start));
    wmmaGemm<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    float basicTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&basicTime, start, stop));
    float basicGflops = 2.0 * M * N * K / basicTime / 1e6;

    printf("Time: %.3f ms\\n", basicTime);
    printf("Performance: %.2f GFLOPS\\n\\n", basicGflops);

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost));

    // --- Test 2: Tiled WMMA GEMM ---
    printf("Test 2: Tiled WMMA GEMM\\n");

    CUDA_CHECK(cudaMemset(d_C, 0, bytesC));

    const int BLOCK_TILES = 4;
    dim3 tiledBlockDim(128);
    dim3 tiledGridDim((M + WMMA_M * BLOCK_TILES - 1) / (WMMA_M * BLOCK_TILES),
                      (N + WMMA_N * BLOCK_TILES - 1) / (WMMA_N * BLOCK_TILES));

    printf("Grid: (%d, %d), Block: %d\\n", tiledGridDim.x, tiledGridDim.y, tiledBlockDim.x);

    // Warm-up
    wmmaGemmTiled<<<tiledGridDim, tiledBlockDim>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed run
    CUDA_CHECK(cudaEventRecord(start));
    wmmaGemmTiled<<<tiledGridDim, tiledBlockDim>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    float tiledTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&tiledTime, start, stop));
    float tiledGflops = 2.0 * M * N * K / tiledTime / 1e6;

    printf("Time: %.3f ms\\n", tiledTime);
    printf("Performance: %.2f GFLOPS\\n", tiledGflops);
    printf("Speedup vs basic: %.2fx\\n\\n", basicTime / tiledTime);

    // Verify results
    printf("Verifying results...\\n");
    cpuGemm(h_A, h_B, h_C_ref, M, N, K, alpha, beta);

    float maxError = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float error = fabsf(h_C[i] - h_C_ref[i]);
        maxError = fmaxf(maxError, error);
    }

    printf("Max error: %f\\n", maxError);
    if (maxError < 1e-2) {
        printf("Verification: PASSED\\n\\n");
    } else {
        printf("Verification: FAILED\\n\\n");
    }

    // Summary
    printf("=== Performance Summary ===\\n");
    printf("Basic WMMA: %.2f GFLOPS\\n", basicGflops);
    printf("Tiled WMMA: %.2f GFLOPS\\n\\n", tiledGflops);

    printf("=== Key Takeaways ===\\n");
    printf("1. WMMA API provides direct access to Tensor Cores\\n");
    printf("2. Operations are performed on 16x16x16 matrix fragments\\n");
    printf("3. Mixed precision (FP16 input, FP32 accumulation) is standard\\n");
    printf("4. Tiling improves performance through better data reuse\\n");
    printf("5. For production, use cuBLAS which is highly optimized\\n");

    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    free(h_A_fp16);
    free(h_B_fp16);

    return 0;
#endif
}
"""
}

def main():
    """Generate all CUDA sample files."""
    print(f"Generating Phase 9 CUDA samples in {OUTPUT_DIR}/")
    print("=" * 60)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate each file
    for filename, content in CUDA_SAMPLES.items():
        filepath = os.path.join(OUTPUT_DIR, filename)
        print(f"\nCreating {filename}...")

        with open(filepath, 'w') as f:
            f.write(content)

        print(f"  ✓ Written to {filepath}")
        print(f"  Lines: {len(content.splitlines())}")
        print(f"  Size: {len(content)} bytes")

    print("\n" + "=" * 60)
    print(f"Successfully generated {len(CUDA_SAMPLES)} files!")
    print("\nFiles created:")
    for filename in CUDA_SAMPLES.keys():
        print(f"  - {OUTPUT_DIR}/{filename}")

    print("\nCompilation examples:")
    print("  nvcc -arch=sm_35 -rdc=true 50_dynamic_parallelism.cu -o dynamic_parallelism")
    print("  nvcc -arch=sm_70 51_cuda_graphs.cu -o cuda_graphs")
    print("  nvcc -arch=sm_70 52_mps_demo.cu -o mps_demo")
    print("  nvcc -arch=sm_70 53_mixed_precision.cu -o mixed_precision")
    print("  nvcc -arch=sm_70 54_tensor_cores.cu -o tensor_cores")
    print("  nvcc -arch=sm_70 55_wmma_gemm.cu -o wmma_gemm")

    print("\nNote: Tensor Cores and WMMA require sm_70 (Volta) or higher")
    print("      Dynamic Parallelism requires sm_35 (Kepler) or higher")

if __name__ == "__main__":
    main()
