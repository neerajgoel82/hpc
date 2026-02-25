#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
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

__global__ void monteCarloOptionPricing(float *prices, int numPaths, int numSteps,
                                         float S0, float K, float r, float sigma,
                                         float T, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPaths) return;

    // Initialize random number generator
    curandState state;
    curand_init(seed, idx, 0, &state);

    float dt = T / numSteps;
    float S = S0;

    // Simulate price path
    for (int step = 0; step < numSteps; step++) {
        float z = curand_normal(&state);
        S *= expf((r - 0.5f * sigma * sigma) * dt + sigma * sqrtf(dt) * z);
    }

    // Calculate payoff for European call option
    prices[idx] = fmaxf(S - K, 0.0f);
}

int main() {
    printf("=== Monte Carlo Option Pricing ===\n\n");

    // Option parameters
    const float S0 = 100.0f;      // Initial stock price
    const float K = 105.0f;       // Strike price
    const float r = 0.05f;        // Risk-free rate (5%)
    const float sigma = 0.2f;     // Volatility (20%)
    const float T = 1.0f;         // Time to maturity (1 year)

    const int numPaths = 1000000;  // Number of Monte Carlo paths
    const int numSteps = 252;      // Trading days in a year

    printf("European Call Option Parameters:\n");
    printf("  Spot price (S0):    $%.2f\n", S0);
    printf("  Strike price (K):   $%.2f\n", K);
    printf("  Risk-free rate (r): %.1f%%\n", r * 100);
    printf("  Volatility (σ):     %.1f%%\n", sigma * 100);
    printf("  Time to maturity:   %.1f years\n", T);
    printf("\nMonte Carlo simulation:\n");
    printf("  Paths: %d\n", numPaths);
    printf("  Steps per path: %d\n\n", numSteps);

    // Allocate device memory
    float *d_prices;
    CUDA_CHECK(cudaMalloc(&d_prices, numPaths * sizeof(float)));

    int threads = 256;
    int blocks = (numPaths + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    monteCarloOptionPricing<<<blocks, threads>>>
        (d_prices, numPaths, numSteps, S0, K, r, sigma, T, 1234ULL);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // Copy results back
    float *h_prices = (float*)malloc(numPaths * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_prices, d_prices, numPaths * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Calculate option price (discounted expected payoff)
    double sum = 0.0;
    for (int i = 0; i < numPaths; i++) {
        sum += h_prices[i];
    }
    float optionPrice = expf(-r * T) * (sum / numPaths);

    // Black-Scholes formula for comparison
    float d1 = (logf(S0/K) + (r + 0.5f*sigma*sigma)*T) / (sigma * sqrtf(T));
    float d2 = d1 - sigma * sqrtf(T);

    // Approximate normal CDF using error function
    auto normcdf = [](float x) {
        return 0.5f * (1.0f + erff(x / sqrtf(2.0f)));
    };

    float bsPrice = S0 * normcdf(d1) - K * expf(-r*T) * normcdf(d2);

    printf("Results:\n");
    printf("  Monte Carlo price: $%.4f\n", optionPrice);
    printf("  Black-Scholes price: $%.4f\n", bsPrice);
    printf("  Difference: $%.4f (%.2f%%)\n",
           fabs(optionPrice - bsPrice),
           fabs(optionPrice - bsPrice) / bsPrice * 100);

    printf("\nPerformance:\n");
    printf("  Computation time: %.2f ms\n", ms);
    printf("  Paths/sec: %.2f million\n", numPaths / (ms / 1000.0) / 1e6);

    free(h_prices);
    cudaFree(d_prices);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
