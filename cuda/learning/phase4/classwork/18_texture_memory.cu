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

// Modern texture object API (CUDA 11+)
__global__ void textureKernel(cudaTextureObject_t texObj, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float value = tex2D<float>(texObj, x + 0.5f, y + 0.5f);
        output[y * width + x] = value;
    }
}

int main() {
    printf("=== Texture Memory ===\n\n");
    const int WIDTH = 1024, HEIGHT = 1024;
    size_t bytes = WIDTH * HEIGHT * sizeof(float);

    float *h_input = (float*)malloc(bytes);
    for (int i = 0; i < WIDTH * HEIGHT; i++) h_input[i] = (float)i;

    float *d_output;
    CUDA_CHECK(cudaMalloc(&d_output, bytes));

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray* cuArray;
    CUDA_CHECK(cudaMallocArray(&cuArray, &channelDesc, WIDTH, HEIGHT));
    CUDA_CHECK(cudaMemcpyToArray(cuArray, 0, 0, h_input, bytes, cudaMemcpyHostToDevice));

    // Create texture object
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;

    cudaTextureObject_t texObj = 0;
    CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

    dim3 threads(16, 16);
    dim3 blocks((WIDTH + 15) / 16, (HEIGHT + 15) / 16);

    textureKernel<<<blocks, threads>>>(texObj, d_output, WIDTH, HEIGHT);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Texture memory kernel executed successfully\n");
    printf("Benefits: cached, hardware interpolation, good for 2D access\n");

    CUDA_CHECK(cudaDestroyTextureObject(texObj));
    CUDA_CHECK(cudaFreeArray(cuArray));
    free(h_input);
    cudaFree(d_output);
    return 0;
}
