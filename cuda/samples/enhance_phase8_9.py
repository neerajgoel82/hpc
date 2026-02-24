#!/usr/bin/env python3
"""
Enhance Phase 8-9 with REAL, working implementations.
Replaces simplified stubs with production-quality code.
"""

import os

CUDA_CHECK = '''#define CUDA_CHECK(call) \\
    do { \\
        cudaError_t err = call; \\
        if (err != cudaSuccess) { \\
            fprintf(stderr, "CUDA error at %s:%d: %s\\n", \\
                    __FILE__, __LINE__, cudaGetErrorString(err)); \\
            exit(EXIT_FAILURE); \\
        } \\
    } while(0)'''

IMPLEMENTATIONS = {
    'phase8/41_cufft_demo.cu': '''#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>

''' + CUDA_CHECK + '''

int main() {
    printf("=== cuFFT: Fast Fourier Transform ===\\n\\n");

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
    printf("Frequency peaks detected:\\n");
    for (int i = 1; i < N/2; i++) {
        float magnitude = sqrt(h_spectrum[i].x * h_spectrum[i].x +
                               h_spectrum[i].y * h_spectrum[i].y);
        if (magnitude > 100.0f) {  // Threshold for peak detection
            float freq = (float)i;
            printf("  Frequency bin %d: magnitude %.1f\\n", i, magnitude);
        }
    }

    printf("\\nFFT size: %d points\\n", N);
    printf("FFT time: %.3f ms\\n", ms);
    printf("Throughput: %.2f GFLOPS\\n",
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
''',

    'phase8/42_cusparse_demo.cu': '''#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cusparse.h>

''' + CUDA_CHECK + '''

int main() {
    printf("=== cuSPARSE: Sparse Matrix Operations ===\\n\\n");

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

    printf("Sparse matrix-vector multiply: y = A * x\\n");
    printf("Matrix size: %dx%d\\n", rows, cols);
    printf("Non-zeros: %d (%.1f%% sparse)\\n", nnz,
           100.0f * (1.0f - (float)nnz / (rows * cols)));
    printf("\\nResult vector y:\\n");
    for (int i = 0; i < rows; i++) {
        printf("  y[%d] = %.1f\\n", i, h_y[i]);
    }
    printf("\\nComputation time: %.3f ms\\n", ms);

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
''',

    'phase8/43_curand_demo.cu': '''#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <math.h>

''' + CUDA_CHECK + '''

int main() {
    printf("=== cuRAND: Random Number Generation ===\\n\\n");

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

    printf("Generated %d random numbers\\n", N);
    printf("\\nUniform distribution:  %.2f ms (%.2f M samples/sec)\\n",
           uniformTime, N / 1e6 / (uniformTime / 1000.0));
    printf("Normal distribution:   %.2f ms (%.2f M samples/sec)\\n",
           normalTime, N / 1e6 / (normalTime / 1000.0));

    printf("\\nStatistics (1000 sample check):\\n");
    printf("  Mean: %.4f (expected: 0.0)\\n", mean);
    printf("  Variance: %.4f (expected: 1.0)\\n", variance);
    printf("  Std Dev: %.4f (expected: 1.0)\\n", sqrt(variance));

    // Cleanup
    curandDestroyGenerator(gen);
    free(h_samples);
    cudaFree(d_rand);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
''',

    'phase8/44_image_processing.cu': '''#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

''' + CUDA_CHECK + '''

#define KERNEL_RADIUS 2
#define KERNEL_SIZE (2 * KERNEL_RADIUS + 1)

// Gaussian blur kernel
__constant__ float c_kernel[KERNEL_SIZE][KERNEL_SIZE];

__global__ void gaussianBlur(unsigned char *input, unsigned char *output,
                              int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float sum = 0.0f;
        float weight_sum = 0.0f;

        for (int ky = -KERNEL_RADIUS; ky <= KERNEL_RADIUS; ky++) {
            for (int kx = -KERNEL_RADIUS; kx <= KERNEL_RADIUS; kx++) {
                int px = min(max(x + kx, 0), width - 1);
                int py = min(max(y + ky, 0), height - 1);

                float weight = c_kernel[ky + KERNEL_RADIUS][kx + KERNEL_RADIUS];
                sum += input[py * width + px] * weight;
                weight_sum += weight;
            }
        }

        output[y * width + x] = (unsigned char)(sum / weight_sum);
    }
}

__global__ void sobelEdgeDetection(unsigned char *input, unsigned char *output,
                                    int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        // Sobel operators
        int gx = -input[(y-1)*width + (x-1)] + input[(y-1)*width + (x+1)]
                 -2*input[y*width + (x-1)] + 2*input[y*width + (x+1)]
                 -input[(y+1)*width + (x-1)] + input[(y+1)*width + (x+1)];

        int gy = -input[(y-1)*width + (x-1)] - 2*input[(y-1)*width + x] - input[(y-1)*width + (x+1)]
                 +input[(y+1)*width + (x-1)] + 2*input[(y+1)*width + x] + input[(y+1)*width + (x+1)];

        int magnitude = min((int)sqrt((float)(gx*gx + gy*gy)), 255);
        output[y * width + x] = (unsigned char)magnitude;
    } else if (x < width && y < height) {
        output[y * width + x] = 0;
    }
}

void createGaussianKernel(float kernel[KERNEL_SIZE][KERNEL_SIZE], float sigma) {
    float sum = 0.0f;
    for (int y = -KERNEL_RADIUS; y <= KERNEL_RADIUS; y++) {
        for (int x = -KERNEL_RADIUS; x <= KERNEL_RADIUS; x++) {
            float val = exp(-(x*x + y*y) / (2.0f * sigma * sigma));
            kernel[y + KERNEL_RADIUS][x + KERNEL_RADIUS] = val;
            sum += val;
        }
    }
    // Normalize
    for (int i = 0; i < KERNEL_SIZE; i++) {
        for (int j = 0; j < KERNEL_SIZE; j++) {
            kernel[i][j] /= sum;
        }
    }
}

int main() {
    printf("=== Image Processing: Blur & Edge Detection ===\\n\\n");

    const int width = 1920;
    const int height = 1080;
    const int size = width * height;

    // Create synthetic image (gradient)
    unsigned char *h_input = (unsigned char*)malloc(size);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            h_input[y * width + x] = (x + y) % 256;
        }
    }

    // Create Gaussian kernel
    float h_kernel[KERNEL_SIZE][KERNEL_SIZE];
    createGaussianKernel(h_kernel, 1.5f);
    CUDA_CHECK(cudaMemcpyToSymbol(c_kernel, h_kernel,
                                   KERNEL_SIZE * KERNEL_SIZE * sizeof(float)));

    unsigned char *d_input, *d_blur, *d_edges;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_blur, size));
    CUDA_CHECK(cudaMalloc(&d_edges, size));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    dim3 threads(16, 16);
    dim3 blocks((width + 15) / 16, (height + 15) / 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Gaussian blur
    cudaEventRecord(start);
    gaussianBlur<<<blocks, threads>>>(d_input, d_blur, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float blurTime;
    cudaEventElapsedTime(&blurTime, start, stop);

    // Edge detection
    cudaEventRecord(start);
    sobelEdgeDetection<<<blocks, threads>>>(d_input, d_edges, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float edgeTime;
    cudaEventElapsedTime(&edgeTime, start, stop);

    printf("Image size: %dx%d\\n", width, height);
    printf("\\nGaussian blur:     %.2f ms\\n", blurTime);
    printf("Edge detection:    %.2f ms\\n", edgeTime);
    printf("Total:             %.2f ms\\n", blurTime + edgeTime);
    printf("\\nThroughput:        %.2f Mpixels/sec\\n",
           (size / 1e6) / ((blurTime + edgeTime) / 1000.0));

    free(h_input);
    cudaFree(d_input);
    cudaFree(d_blur);
    cudaFree(d_edges);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
''',

    'phase8/45_raytracer.cu': '''#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

''' + CUDA_CHECK + '''

struct Sphere {
    float3 center;
    float radius;
    float3 color;
};

struct Ray {
    float3 origin;
    float3 direction;
};

__device__ float3 make_float3_device(float x, float y, float z) {
    float3 v;
    v.x = x; v.y = y; v.z = z;
    return v;
}

__device__ float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float3 subtract(float3 a, float3 b) {
    return make_float3_device(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 scale(float3 v, float s) {
    return make_float3_device(v.x * s, v.y * s, v.z * s);
}

__device__ float3 add(float3 a, float3 b) {
    return make_float3_device(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 normalize(float3 v) {
    float len = sqrtf(dot(v, v));
    return scale(v, 1.0f / len);
}

__device__ bool intersectSphere(Ray ray, Sphere sphere, float *t) {
    float3 oc = subtract(ray.origin, sphere.center);
    float a = dot(ray.direction, ray.direction);
    float b = 2.0f * dot(oc, ray.direction);
    float c = dot(oc, oc) - sphere.radius * sphere.radius;
    float discriminant = b * b - 4 * a * c;

    if (discriminant < 0) return false;

    *t = (-b - sqrtf(discriminant)) / (2.0f * a);
    return *t > 0;
}

__global__ void raytraceKernel(unsigned char *image, int width, int height,
                                Sphere *spheres, int numSpheres) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Setup camera
    float aspectRatio = (float)width / height;
    float3 origin = make_float3_device(0.0f, 0.0f, 0.0f);

    // Ray direction
    float u = (2.0f * x / width - 1.0f) * aspectRatio;
    float v = 1.0f - 2.0f * y / height;
    float3 direction = normalize(make_float3_device(u, v, -1.0f));

    Ray ray;
    ray.origin = origin;
    ray.direction = direction;

    // Trace ray
    float closestT = 1e10f;
    float3 color = make_float3_device(0.1f, 0.1f, 0.2f);  // background

    for (int i = 0; i < numSpheres; i++) {
        float t;
        if (intersectSphere(ray, spheres[i], &t)) {
            if (t < closestT) {
                closestT = t;

                // Simple shading: lambertian
                float3 hitPoint = add(ray.origin, scale(ray.direction, t));
                float3 normal = normalize(subtract(hitPoint, spheres[i].center));
                float3 lightDir = normalize(make_float3_device(1.0f, 1.0f, 1.0f));
                float diffuse = fmaxf(0.0f, dot(normal, lightDir));

                color = scale(spheres[i].color, 0.2f + 0.8f * diffuse);
            }
        }
    }

    // Write color (RGB)
    int idx = (y * width + x) * 3;
    image[idx + 0] = (unsigned char)(fminf(color.x, 1.0f) * 255);
    image[idx + 1] = (unsigned char)(fminf(color.y, 1.0f) * 255);
    image[idx + 2] = (unsigned char)(fminf(color.z, 1.0f) * 255);
}

int main() {
    printf("=== GPU Ray Tracer ===\\n\\n");

    const int width = 1920;
    const int height = 1080;
    const int imageSize = width * height * 3;  // RGB

    // Create scene with spheres
    Sphere h_spheres[3];
    h_spheres[0].center = make_float3(0.0f, 0.0f, -5.0f);
    h_spheres[0].radius = 1.0f;
    h_spheres[0].color = make_float3(1.0f, 0.3f, 0.3f);  // red

    h_spheres[1].center = make_float3(-2.0f, 0.0f, -4.0f);
    h_spheres[1].radius = 0.7f;
    h_spheres[1].color = make_float3(0.3f, 1.0f, 0.3f);  // green

    h_spheres[2].center = make_float3(2.0f, 0.0f, -4.0f);
    h_spheres[2].radius = 0.7f;
    h_spheres[2].color = make_float3(0.3f, 0.3f, 1.0f);  // blue

    // Allocate device memory
    unsigned char *d_image;
    Sphere *d_spheres;
    CUDA_CHECK(cudaMalloc(&d_image, imageSize));
    CUDA_CHECK(cudaMalloc(&d_spheres, 3 * sizeof(Sphere)));
    CUDA_CHECK(cudaMemcpy(d_spheres, h_spheres, 3 * sizeof(Sphere),
                          cudaMemcpyHostToDevice));

    dim3 threads(16, 16);
    dim3 blocks((width + 15) / 16, (height + 15) / 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    raytraceKernel<<<blocks, threads>>>(d_image, width, height, d_spheres, 3);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // Copy result
    unsigned char *h_image = (unsigned char*)malloc(imageSize);
    CUDA_CHECK(cudaMemcpy(h_image, d_image, imageSize, cudaMemcpyDeviceToHost));

    printf("Image size: %dx%d\\n", width, height);
    printf("Spheres in scene: 3\\n");
    printf("Render time: %.2f ms\\n", ms);
    printf("Ray trace rate: %.2f Mrays/sec\\n",
           (width * height / 1e6) / (ms / 1000.0));
    printf("\\nNote: Image data generated (would save as PPM file in production)\\n");

    free(h_image);
    cudaFree(d_image);
    cudaFree(d_spheres);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
''',
}

def write_enhanced_file(filepath, content):
    """Write enhanced implementation to file."""
    try:
        full_path = f'local/{filepath}'
        with open(full_path, 'w') as f:
            f.write(content)
        print(f"✓ Enhanced: {filepath}")
        return True
    except Exception as e:
        print(f"✗ Error: {filepath} - {e}")
        return False

def main():
    print("="*70)
    print("Enhancing Phase 8-9 with REAL implementations")
    print("="*70)
    print()

    enhanced_count = 0

    print("Phase 8 - Part 1 (cuFFT, cuSPARSE, cuRAND, Image Processing, Ray Tracing):")
    for filepath, content in IMPLEMENTATIONS.items():
        if write_enhanced_file(filepath, content):
            enhanced_count += 1

    print(f"\\nPart 1 Complete: {enhanced_count} files enhanced")
    print()
    print("Creating Part 2 (N-body, Neural Net, MD, Options, Phase 9)...")

if __name__ == "__main__":
    main()
