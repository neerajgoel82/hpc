#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cufft.h>
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
    printf("=== cuFFT: Fast Fourier Transform ===\n\n");

    const int N = 1024;  // Signal length
    const int SIGNAL_SIZE = N * sizeof(cufftComplex);

    // Create input signal: mix of sine waves
    cufftComplex *h_signal = (cufftComplex*)malloc(SIGNAL_SIZE);
    for (int i = 0; i < N; i++) {
        float t = (float)i / N;
        // Mix of 50 Hz and 120 Hz signals
        h_signal[i].x = cos(2.0f * M_PI * 50.0f * t) +
                        0.5f * cos(2.0f * M_PI * 120.0f * t);  // real
        h_signal[i].y = 0.0f;  // imaginary
    }

    // Allocate device memory
    cufftComplex *d_signal;
    CUDA_CHECK(cudaMalloc(&d_signal, SIGNAL_SIZE));
    CUDA_CHECK(cudaMemcpy(d_signal, h_signal, SIGNAL_SIZE, cudaMemcpyHostToDevice));

    // Create cuFFT plan
    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_C2C, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Execute FFT (forward transform)
    cudaEventRecord(start);
    cufftExecC2C(plan, d_signal, d_signal, CUFFT_FORWARD);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // Copy result back
    cufftComplex *h_spectrum = (cufftComplex*)malloc(SIGNAL_SIZE);
    CUDA_CHECK(cudaMemcpy(h_spectrum, d_signal, SIGNAL_SIZE, cudaMemcpyDeviceToHost));

    // Find peaks in frequency spectrum
    printf("Frequency peaks detected:\n");
    for (int i = 1; i < N/2; i++) {
        float magnitude = sqrt(h_spectrum[i].x * h_spectrum[i].x +
                               h_spectrum[i].y * h_spectrum[i].y);
        if (magnitude > 100.0f) {  // Threshold for peak detection
            float freq = (float)i;
            printf("  Frequency bin %d: magnitude %.1f\n", i, magnitude);
        }
    }

    printf("\nFFT size: %d points\n", N);
    printf("FFT time: %.3f ms\n", ms);
    printf("Throughput: %.2f GFLOPS\n",
           (5.0 * N * log2(N) / 1e9) / (ms / 1000.0));

    // Cleanup
    cufftDestroy(plan);
    free(h_signal);
    free(h_spectrum);
    cudaFree(d_signal);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
