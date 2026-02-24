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


texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;

__global__ void textureKernel(float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float value = tex2D(texRef, x + 0.5f, y + 0.5f);
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

    texRef.addressMode[0] = cudaAddressModeClamp;
    texRef.addressMode[1] = cudaAddressModeClamp;
    texRef.filterMode = cudaFilterModeLinear;
    texRef.normalized = false;

    CUDA_CHECK(cudaBindTextureToArray(texRef, cuArray, channelDesc));

    dim3 threads(16, 16);
    dim3 blocks((WIDTH + 15) / 16, (HEIGHT + 15) / 16);

    textureKernel<<<blocks, threads>>>(d_output, WIDTH, HEIGHT);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Texture memory kernel executed successfully\n");
    printf("Benefits: cached, hardware interpolation, good for 2D access\n");

    CUDA_CHECK(cudaUnbindTexture(texRef));
    CUDA_CHECK(cudaFreeArray(cuArray));
    free(h_input);
    cudaFree(d_output);
    return 0;
}
