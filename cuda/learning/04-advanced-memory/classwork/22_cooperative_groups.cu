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


#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void cooperativeKernel(float* data, int n) {
    cg::thread_block block = cg::this_thread_block();
    int idx = block.group_index().x * block.group_dim().x + block.thread_rank();

    if (idx < n) data[idx] *= 2.0f;
    block.sync();

    cg::thread_block_tile<32> tile = cg::tiled_partition<32>(block);
    if (idx < n) {
        float val = data[idx];
        for (int offset = tile.size() / 2; offset > 0; offset /= 2)
            val += tile.shfl_down(val, offset);

        if (tile.thread_rank() == 0 && idx / 32 < n / 32)
            data[idx] = val;
    }
}

int main() {
    printf("=== Cooperative Groups ===\n\n");
    const int N = 256;
    size_t bytes = N * sizeof(float);

    float *h_data = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_data[i] = (float)(i + 1);

    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    cooperativeKernel<<<(N + 127) / 128, 128>>>(d_data, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));
    printf("Cooperative kernel executed\n");
    printf("First warp result: %.0f\n", h_data[0]);
    printf("Features: flexible grouping, warp primitives, grid-sync\n");

    free(h_data);
    cudaFree(d_data);
    return 0;
}
