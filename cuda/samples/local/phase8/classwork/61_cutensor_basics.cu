/*
 * cuTENSOR Basics - High-performance tensor operations
 *
 * cuTENSOR is NVIDIA's library for tensor contractions and operations.
 * Essential for scientific computing, quantum chemistry, and AI/ML workloads.
 *
 * Key Operations:
 * - Tensor contractions (Einstein notation)
 * - Element-wise operations (add, multiply, etc.)
 * - Tensor permutations (transpose)
 * - Reductions (sum, max, min)
 *
 * Why cuTENSOR:
 * - Optimized for modern GPU architectures (Tensor Cores)
 * - Handles arbitrary tensor dimensions and sizes
 * - Supports mixed precision (FP16, FP32, TF32, etc.)
 * - Much faster than manual implementations
 *
 * Requires: CUDA 10.1+, cuTENSOR library
 * Compile: nvcc -arch=sm_70 -O2 61_cutensor_basics.cu -o cutensor_basics -lcutensor
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cutensor.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUTENSOR_CHECK(call) \
    do { \
        cutensorStatus_t err = call; \
        if (err != CUTENSOR_STATUS_SUCCESS) { \
            fprintf(stderr, "cuTENSOR error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cutensorGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Example 1: Simple tensor contraction (matrix multiplication)
// C[m,n] = A[m,k] * B[k,n]
void tensorContraction() {
    printf("\n=== Tensor Contraction (Matrix Multiply) ===\n");

    // Matrix dimensions
    int m = 256, n = 256, k = 256;

    // Allocate host memory
    float *h_A = (float*)malloc(m * k * sizeof(float));
    float *h_B = (float*)malloc(k * n * sizeof(float));
    float *h_C = (float*)malloc(m * n * sizeof(float));

    // Initialize matrices
    for (int i = 0; i < m * k; i++) h_A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < k * n; i++) h_B[i] = (float)rand() / RAND_MAX;

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, m * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, k * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, m * n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice));

    // Initialize cuTENSOR
    cutensorHandle_t handle;
    CUTENSOR_CHECK(cutensorInit(&handle));

    // Define tensor descriptors
    // A[m,k] - modes: 'm', 'k'
    // B[k,n] - modes: 'k', 'n'
    // C[m,n] - modes: 'm', 'n'

    cutensorTensorDescriptor_t descA, descB, descC;

    // Extents (dimensions)
    int64_t extentA[] = {m, k};
    int64_t extentB[] = {k, n};
    int64_t extentC[] = {m, n};

    // Strides (column-major)
    int64_t strideA[] = {1, m};
    int64_t strideB[] = {1, k};
    int64_t strideC[] = {1, m};

    // Modes (dimension labels)
    int32_t modeA[] = {'m', 'k'};
    int32_t modeB[] = {'k', 'n'};
    int32_t modeC[] = {'m', 'n'};

    // Create tensor descriptors
    CUTENSOR_CHECK(cutensorInitTensorDescriptor(
        &handle, &descA, 2, extentA, strideA,
        CUDA_R_32F, CUTENSOR_OP_IDENTITY));

    CUTENSOR_CHECK(cutensorInitTensorDescriptor(
        &handle, &descB, 2, extentB, strideB,
        CUDA_R_32F, CUTENSOR_OP_IDENTITY));

    CUTENSOR_CHECK(cutensorInitTensorDescriptor(
        &handle, &descC, 2, extentC, strideC,
        CUDA_R_32F, CUTENSOR_OP_IDENTITY));

    // Set up contraction descriptor
    // C[m,n] = A[m,k] * B[k,n]
    // In Einstein notation: C_mn = A_mk * B_kn
    cutensorContractionDescriptor_t desc;
    CUTENSOR_CHECK(cutensorInitContractionDescriptor(
        &handle, &desc,
        &descA, modeA, /* unary operator A */ CUTENSOR_OP_IDENTITY,
        &descB, modeB, /* unary operator B */ CUTENSOR_OP_IDENTITY,
        &descC, modeC, /* unary operator C */ CUTENSOR_OP_IDENTITY,
        &descC, modeC,
        CUTENSOR_COMPUTE_32F));

    // Find best algorithm
    cutensorContractionFind_t find;
    CUTENSOR_CHECK(cutensorInitContractionFind(&handle, &find, CUTENSOR_ALGO_DEFAULT));

    // Query workspace size
    uint64_t worksize = 0;
    CUTENSOR_CHECK(cutensorContractionGetWorkspaceSize(
        &handle, &desc, &find, CUTENSOR_WORKSPACE_RECOMMENDED, &worksize));

    void *work = nullptr;
    if (worksize > 0) {
        CUDA_CHECK(cudaMalloc(&work, worksize));
    }

    // Set up contraction plan
    cutensorContractionPlan_t plan;
    CUTENSOR_CHECK(cutensorInitContractionPlan(&handle, &plan, &desc, &find, worksize));

    // Execute contraction: C = alpha * A * B + beta * C
    float alpha = 1.0f;
    float beta = 0.0f;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    CUTENSOR_CHECK(cutensorContraction(
        &handle, &plan,
        &alpha, d_A, d_B,
        &beta, d_C, d_C,
        work, worksize, 0 /* stream */));

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    printf("Matrix size: %dx%dx%d\n", m, k, n);
    printf("Time: %.3f ms\n", ms);
    printf("GFLOPS: %.2f\n", (2.0 * m * n * k) / (ms * 1e6));

    // Verify result (spot check)
    CUDA_CHECK(cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost));

    // Simple verification: C[0,0] = sum(A[0,:] * B[:,0])
    float expected = 0.0f;
    for (int i = 0; i < k; i++) {
        expected += h_A[i * m] * h_B[i];
    }
    float result = h_C[0];
    printf("Verification: C[0,0] = %.6f (expected: %.6f, error: %.2e)\n",
           result, expected, fabsf(result - expected));

    // Cleanup
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    if (work) cudaFree(work);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Example 2: Element-wise operations
void elementwiseOps() {
    printf("\n=== Element-wise Operations ===\n");

    int n = 1024 * 1024;

    float *h_A = (float*)malloc(n * sizeof(float));
    float *h_B = (float*)malloc(n * sizeof(float));
    float *h_C = (float*)malloc(n * sizeof(float));

    for (int i = 0; i < n; i++) {
        h_A[i] = (float)i / 1000.0f;
        h_B[i] = (float)(i % 100) / 10.0f;
    }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, n * sizeof(float), cudaMemcpyHostToDevice));

    cutensorHandle_t handle;
    CUTENSOR_CHECK(cutensorInit(&handle));

    // 1D tensor descriptors
    int64_t extent[] = {n};
    int64_t stride[] = {1};

    cutensorTensorDescriptor_t descA, descB, descC;

    CUTENSOR_CHECK(cutensorInitTensorDescriptor(
        &handle, &descA, 1, extent, stride,
        CUDA_R_32F, CUTENSOR_OP_IDENTITY));

    CUTENSOR_CHECK(cutensorInitTensorDescriptor(
        &handle, &descB, 1, extent, stride,
        CUDA_R_32F, CUTENSOR_OP_IDENTITY));

    CUTENSOR_CHECK(cutensorInitTensorDescriptor(
        &handle, &descC, 1, extent, stride,
        CUDA_R_32F, CUTENSOR_OP_IDENTITY));

    // Element-wise addition: C = alpha * A + beta * B
    float alpha = 1.0f;
    float beta = 2.0f;

    int32_t mode[] = {'x'};

    CUTENSOR_CHECK(cutensorElementwiseBinary(
        &handle,
        &alpha, d_A, &descA, mode,
        &beta, d_B, &descB, mode,
        d_C, &descC, mode,
        CUTENSOR_OP_ADD, CUTENSOR_COMPUTE_32F, 0));

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C, d_C, n * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify
    float expected = alpha * h_A[100] + beta * h_B[100];
    printf("Element-wise add: C[100] = %.3f (expected: %.3f)\n", h_C[100], expected);

    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

// Example 3: Tensor permutation (transpose)
void tensorPermutation() {
    printf("\n=== Tensor Permutation (Transpose) ===\n");

    int m = 512, n = 512;

    float *h_A = (float*)malloc(m * n * sizeof(float));
    float *h_B = (float*)malloc(m * n * sizeof(float));

    for (int i = 0; i < m * n; i++) {
        h_A[i] = (float)i;
    }

    float *d_A, *d_B;
    CUDA_CHECK(cudaMalloc(&d_A, m * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, m * n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, m * n * sizeof(float), cudaMemcpyHostToDevice));

    cutensorHandle_t handle;
    CUTENSOR_CHECK(cutensorInit(&handle));

    // A[m,n] -> B[n,m] (transpose)
    int64_t extentA[] = {m, n};
    int64_t strideA[] = {1, m};  // Column-major

    int64_t extentB[] = {n, m};
    int64_t strideB[] = {1, n};  // Column-major

    cutensorTensorDescriptor_t descA, descB;

    CUTENSOR_CHECK(cutensorInitTensorDescriptor(
        &handle, &descA, 2, extentA, strideA,
        CUDA_R_32F, CUTENSOR_OP_IDENTITY));

    CUTENSOR_CHECK(cutensorInitTensorDescriptor(
        &handle, &descB, 2, extentB, strideB,
        CUDA_R_32F, CUTENSOR_OP_IDENTITY));

    // Permutation: B[n,m] = A[m,n]
    int32_t modeA[] = {'m', 'n'};
    int32_t modeB[] = {'n', 'm'};

    float alpha = 1.0f;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    CUTENSOR_CHECK(cutensorPermutation(
        &handle,
        &alpha, d_A, &descA, modeA,
        d_B, &descB, modeB,
        CUTENSOR_COMPUTE_32F, 0));

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    printf("Transpose %dx%d matrix\n", m, n);
    printf("Time: %.3f ms\n", ms);
    printf("Bandwidth: %.2f GB/s\n",
           (2.0 * m * n * sizeof(float)) / (ms * 1e6));

    CUDA_CHECK(cudaMemcpy(h_B, d_B, m * n * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify: B[j,i] should equal A[i,j]
    int i = 10, j = 20;
    float a_val = h_A[j * m + i];  // A[i,j] in column-major
    float b_val = h_B[i * n + j];  // B[j,i] in column-major
    printf("Verification: A[%d,%d] = %.1f, B[%d,%d] = %.1f\n",
           i, j, a_val, j, i, b_val);

    free(h_A); free(h_B);
    cudaFree(d_A); cudaFree(d_B);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    printf("=== cuTENSOR Basics Demo ===\n");

    // Check device
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("\nDevice: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);

    // Get cuTENSOR version
    cutensorHandle_t temp_handle;
    CUTENSOR_CHECK(cutensorInit(&temp_handle));
    size_t version;
    CUTENSOR_CHECK(cutensorGetVersion(&temp_handle, &version));
    printf("cuTENSOR version: %zu\n", version);

    // Run examples
    tensorContraction();
    elementwiseOps();
    tensorPermutation();

    printf("\n=== Key Concepts ===\n");
    printf("1. Tensor contractions use Einstein notation (e.g., C_mn = A_mk * B_kn)\n");
    printf("2. cuTENSOR optimizes for modern GPU architectures (Tensor Cores)\n");
    printf("3. Handles arbitrary tensor dimensions and permutations\n");
    printf("4. Essential for quantum chemistry, physics simulations, AI/ML\n");
    printf("5. Much faster than manual CUDA implementations\n");

    printf("\n=== Applications ===\n");
    printf("- Quantum chemistry (coupled cluster methods)\n");
    printf("- Physics simulations (lattice QCD)\n");
    printf("- Machine learning (tensor networks)\n");
    printf("- Scientific computing (multi-dimensional FFTs)\n");

    return 0;
}
