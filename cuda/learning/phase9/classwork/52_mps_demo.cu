
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

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
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
    printf("=== MPS Status Check ===\n\n");

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
            printf("MPS status: Not detected or not running\n");
        }
    } else {
        printf("Could not check MPS status (nvidia-smi not available)\n");
    }

    printf("\n");
}

void printMPSInfo() {
    printf("=== About CUDA Multi-Process Service (MPS) ===\n\n");

    printf("What is MPS?\n");
    printf("  MPS is a client-server runtime that allows multiple processes\n");
    printf("  to share a single GPU context. This enables better GPU utilization\n");
    printf("  when multiple small workloads run concurrently.\n\n");

    printf("Benefits:\n");
    printf("  - Reduced context switching overhead\n");
    printf("  - Better GPU utilization for small kernels\n");
    printf("  - Multiple processes can use GPU simultaneously\n");
    printf("  - Lower latency for concurrent workloads\n\n");

    printf("When to use MPS:\n");
    printf("  - Multiple MPI ranks per node\n");
    printf("  - Multiple small applications sharing GPU\n");
    printf("  - HPC workloads with many processes\n");
    printf("  - Microservices architecture\n\n");

    printf("How to enable MPS:\n");
    printf("  1. Export CUDA_VISIBLE_DEVICES=0 (or your GPU ID)\n");
    printf("  2. Start MPS daemon: nvidia-cuda-mps-control -d\n");
    printf("  3. Run your applications\n");
    printf("  4. Stop MPS: echo quit | nvidia-cuda-mps-control\n\n");

    printf("Limitations:\n");
    printf("  - Requires Volta or newer for full features\n");
    printf("  - Limited debugger support\n");
    printf("  - All processes must be from same user\n");
    printf("  - Some features require privileged access\n\n");
}

int main(int argc, char **argv) {
    printf("=== CUDA MPS Demo ===\n\n");

    // Get process ID for identification
    int pid = getpid();
    int processNum = 0;

    if (argc > 1) {
        processNum = atoi(argv[1]);
    }

    printf("Process ID: %d\n", pid);
    printf("Process Number: %d\n\n", processNum);

    // Check device properties
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Multi-Process Service: %s\n\n",
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

    printf("Running workload for process %d...\n", processNum);
    printf("Matrix size: %d x %d\n", N, N);
    printf("Grid: %d blocks, Block: %d threads\n\n", blocksPerGrid, threadsPerBlock);

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

    printf("Completed %d iterations\n", iterations);
    printf("Total time: %.3f ms\n", ms);
    printf("Average time per iteration: %.3f ms\n\n", ms / iterations);

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

        printf("=== To Test MPS ===\n\n");
        printf("Run without MPS:\n");
        printf("  Terminal 1: ./mps_demo 0\n");
        printf("  Terminal 2: ./mps_demo 1\n");
        printf("  (Processes will time-slice the GPU)\n\n");

        printf("Run with MPS:\n");
        printf("  1. nvidia-cuda-mps-control -d\n");
        printf("  2. Terminal 1: ./mps_demo 0\n");
        printf("  3. Terminal 2: ./mps_demo 1\n");
        printf("  4. echo quit | nvidia-cuda-mps-control\n");
        printf("  (Processes will share GPU simultaneously)\n\n");

        printf("Compare execution times - MPS should show better total throughput\n");
        printf("when running multiple processes concurrently.\n");
    }

    return 0;
}
