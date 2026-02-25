#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand.h>
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

int main() {
    printf("=== cuRAND: Random Number Generation ===\n\n");

    const int N = 10000000;  // 10M random numbers
    size_t size = N * sizeof(float);

    // Allocate device memory
    float *d_rand;
    CUDA_CHECK(cudaMalloc(&d_rand, size));

    // Create random number generator
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Generate uniform random numbers [0, 1)
    cudaEventRecord(start);
    curandGenerateUniform(gen, d_rand, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float uniformTime;
    cudaEventElapsedTime(&uniformTime, start, stop);

    // Generate normal distribution (mean=0, stddev=1)
    cudaEventRecord(start);
    curandGenerateNormal(gen, d_rand, N, 0.0f, 1.0f);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float normalTime;
    cudaEventElapsedTime(&normalTime, start, stop);

    // Copy some samples back for verification
    float *h_samples = (float*)malloc(1000 * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_samples, d_rand, 1000 * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Calculate statistics
    double sum = 0.0, sum2 = 0.0;
    for (int i = 0; i < 1000; i++) {
        sum += h_samples[i];
        sum2 += h_samples[i] * h_samples[i];
    }
    double mean = sum / 1000.0;
    double variance = sum2 / 1000.0 - mean * mean;

    printf("Generated %d random numbers\n", N);
    printf("\nUniform distribution:  %.2f ms (%.2f M samples/sec)\n",
           uniformTime, N / 1e6 / (uniformTime / 1000.0));
    printf("Normal distribution:   %.2f ms (%.2f M samples/sec)\n",
           normalTime, N / 1e6 / (normalTime / 1000.0));

    printf("\nStatistics (1000 sample check):\n");
    printf("  Mean: %.4f (expected: 0.0)\n", mean);
    printf("  Variance: %.4f (expected: 1.0)\n", variance);
    printf("  Std Dev: %.4f (expected: 1.0)\n", sqrt(variance));

    // Cleanup
    curandDestroyGenerator(gen);
    free(h_samples);
    cudaFree(d_rand);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
