#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cusparse.h>

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
    printf("=== cuSPARSE: Sparse Matrix Operations ===\n\n");

    // Create a sparse matrix in CSR format
    // Matrix: 4x4 with only 6 non-zero elements
    //   [1  0  2  0]
    //   [0  3  0  0]
    //   [0  0  4  5]
    //   [6  0  0  7]

    const int rows = 4;
    const int cols = 4;
    const int nnz = 7;  // number of non-zeros

    // CSR format
    int h_csrRowPtr[5] = {0, 2, 3, 5, 7};  // row pointers
    int h_csrColInd[7] = {0, 2, 1, 2, 3, 0, 3};  // column indices
    float h_csrVal[7] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};  // values

    // Dense vector for multiplication
    float h_x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float h_y[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Allocate device memory
    int *d_csrRowPtr, *d_csrColInd;
    float *d_csrVal, *d_x, *d_y;

    CUDA_CHECK(cudaMalloc(&d_csrRowPtr, (rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_csrColInd, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_csrVal, nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, rows * sizeof(float)));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_csrRowPtr, h_csrRowPtr, (rows + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csrColInd, h_csrColInd, nnz * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csrVal, h_csrVal, nnz * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, cols * sizeof(float), cudaMemcpyHostToDevice));

    // Create cuSPARSE handle
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    // Create matrix descriptor
    cusparseMatDescr_t descr;
    cusparseCreateMatDescr(&descr);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    // Sparse matrix-vector multiplication: y = A * x
    float alpha = 1.0f;
    float beta = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                   rows, cols, nnz, &alpha, descr,
                   d_csrVal, d_csrRowPtr, d_csrColInd,
                   d_x, &beta, d_y);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_y, d_y, rows * sizeof(float), cudaMemcpyDeviceToHost));

    printf("Sparse matrix-vector multiply: y = A * x\n");
    printf("Matrix size: %dx%d\n", rows, cols);
    printf("Non-zeros: %d (%.1f%% sparse)\n", nnz,
           100.0f * (1.0f - (float)nnz / (rows * cols)));
    printf("\nResult vector y:\n");
    for (int i = 0; i < rows; i++) {
        printf("  y[%d] = %.1f\n", i, h_y[i]);
    }
    printf("\nComputation time: %.3f ms\n", ms);

    // Cleanup
    cusparseDestroyMatDescr(descr);
    cusparseDestroy(handle);
    cudaFree(d_csrRowPtr);
    cudaFree(d_csrColInd);
    cudaFree(d_csrVal);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
