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
    printf("=== Image Processing: Blur & Edge Detection ===\n\n");

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

    printf("Image size: %dx%d\n", width, height);
    printf("\nGaussian blur:     %.2f ms\n", blurTime);
    printf("Edge detection:    %.2f ms\n", edgeTime);
    printf("Total:             %.2f ms\n", blurTime + edgeTime);
    printf("\nThroughput:        %.2f Mpixels/sec\n",
           (size / 1e6) / ((blurTime + edgeTime) / 1000.0));

    free(h_input);
    cudaFree(d_input);
    cudaFree(d_blur);
    cudaFree(d_edges);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
