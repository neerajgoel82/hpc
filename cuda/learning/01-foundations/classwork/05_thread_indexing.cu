#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printThreadInfo() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < 10) {
        printf("Global idx=%d: Block(%d,%d,%d) Thread(%d,%d,%d)\n",
               idx,
               blockIdx.x, blockIdx.y, blockIdx.z,
               threadIdx.x, threadIdx.y, threadIdx.z);
    }
}

__global__ void print2DThreadInfo() {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 4 && y < 4) {
        printf("2D [%d,%d]: Block(%d,%d) Thread(%d,%d)\n",
               x, y,
               blockIdx.x, blockIdx.y,
               threadIdx.x, threadIdx.y);
    }
}

__global__ void gridStrideLoop(float *data, int n) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += blockDim.x * gridDim.x) {
        data[idx] = idx * 2.0f;
    }
}

int main() {
    printf("=== Thread Indexing Patterns ===\n\n");

    // 1D indexing
    printf("1D Thread Indexing:\n");
    printThreadInfo<<<2, 8>>>();
    cudaDeviceSynchronize();

    printf("\n2D Thread Indexing:\n");
    dim3 threads2d(2, 2);
    dim3 blocks2d(2, 2);
    print2DThreadInfo<<<blocks2d, threads2d>>>();
    cudaDeviceSynchronize();

    // Grid-stride loop demonstration
    printf("\nGrid-Stride Loop:\n");
    int n = 1000;
    float *d_data;
    cudaMalloc(&d_data, n * sizeof(float));

    // Use fewer blocks than elements
    gridStrideLoop<<<4, 32>>>(d_data, n);
    cudaDeviceSynchronize();

    float h_data[10];
    cudaMemcpy(h_data, d_data, 10 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("First 10 elements: ");
    for (int i = 0; i < 10; i++) {
        printf("%.0f ", h_data[i]);
    }
    printf("\n");

    cudaFree(d_data);

    return 0;
}
