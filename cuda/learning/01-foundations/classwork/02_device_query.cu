#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    printf("=== CUDA Device Query ===\n\n");

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    printf("Number of CUDA devices: %d\n\n", deviceCount);

    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        printf("Device %d: %s\n", dev, prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total global memory: %.2f GB\n", prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
        printf("  Shared memory per block: %zu KB\n", prop.sharedMemPerBlock / 1024);
        printf("  Registers per block: %d\n", prop.regsPerBlock);
        printf("  Warp size: %d\n", prop.warpSize);
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max threads dimensions: (%d, %d, %d)\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Max grid dimensions: (%d, %d, %d)\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("  Clock rate: %.2f GHz\n", prop.clockRate / 1e6);
        printf("  Memory clock rate: %.2f GHz\n", prop.memoryClockRate / 1e6);
        printf("  Memory bus width: %d-bit\n", prop.memoryBusWidth);
        printf("  L2 cache size: %d KB\n", prop.l2CacheSize / 1024);
        printf("  Max constant memory: %zu KB\n", prop.totalConstMem / 1024);
        printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("  Concurrent kernels: %s\n", prop.concurrentKernels ? "Yes" : "No");
        printf("  ECC enabled: %s\n", prop.ECCEnabled ? "Yes" : "No");
        printf("\n");
    }

    return 0;
}
