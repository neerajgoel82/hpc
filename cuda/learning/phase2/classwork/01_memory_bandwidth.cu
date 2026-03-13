// Phase 2: Memory Bandwidth Benchmark
#include <stdio.h>
#include <stdlib.h>
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


float measureBandwidth(size_t size, bool pinned, bool h2d) {
    float *h_data, *d_data;

    if (pinned) {
        CUDA_CHECK(cudaMallocHost(&h_data, size));
    } else {
        h_data = (float*)malloc(size);
    }
    CUDA_CHECK(cudaMalloc(&d_data, size));

    // Initialize
    for (size_t i = 0; i < size/sizeof(float); i++) {
        h_data[i] = i * 1.0f;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    if (h2d) {
        CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float bandwidth = (size / 1e9) / (milliseconds / 1000.0);

    if (pinned) {
        cudaFreeHost(h_data);
    } else {
        free(h_data);
    }
    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return bandwidth;
}

int main() {
    printf("=== Memory Bandwidth Benchmark ===\n\n");

    size_t sizes[] = {1<<20, 1<<22, 1<<24, 1<<26};  // 1MB to 64MB

    printf("%-15s %-20s %-20s %-20s %-20s\n",
           "Size (MB)", "Pageable H→D", "Pinned H→D", "Pageable D→H", "Pinned D→H");
    printf("---------------------------------------------------------------------------------\n");

    for (int i = 0; i < 4; i++) {
        size_t size = sizes[i];
        float bw_page_h2d = measureBandwidth(size, false, true);
        float bw_pin_h2d = measureBandwidth(size, true, true);
        float bw_page_d2h = measureBandwidth(size, false, false);
        float bw_pin_d2h = measureBandwidth(size, true, false);

        printf("%-15zu %-20.2f %-20.2f %-20.2f %-20.2f\n",
               size / (1024*1024), bw_page_h2d, bw_pin_h2d, bw_page_d2h, bw_pin_d2h);
    }

    printf("\nPinned memory provides significantly better bandwidth!\n");
    return 0;
}
