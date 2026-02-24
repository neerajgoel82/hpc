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

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ float sigmoid_derivative(float x) {
    float s = sigmoid(x);
    return s * (1.0f - s);
}

__global__ void forwardLayer(float *input, float *weights, float *bias,
                              float *output, int inputSize, int outputSize) {
    int neuron = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron >= outputSize) return;

    float sum = bias[neuron];
    for (int i = 0; i < inputSize; i++) {
        sum += input[i] * weights[neuron * inputSize + i];
    }
    output[neuron] = sigmoid(sum);
}

__global__ void backwardLayer(float *input, float *weights, float *delta,
                               float *prevDelta, int inputSize, int outputSize) {
    int neuron = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron >= inputSize) return;

    float error = 0.0f;
    for (int i = 0; i < outputSize; i++) {
        error += delta[i] * weights[i * inputSize + neuron];
    }
    prevDelta[neuron] = error * sigmoid_derivative(input[neuron]);
}

__global__ void updateWeights(float *weights, float *input, float *delta,
                               int inputSize, int outputSize, float learningRate) {
    int neuron = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron >= outputSize) return;

    for (int i = 0; i < inputSize; i++) {
        weights[neuron * inputSize + i] += learningRate * delta[neuron] * input[i];
    }
}

int main() {
    printf("=== Neural Network: Forward & Backward Pass ===\n\n");

    const int inputSize = 784;    // 28x28 image
    const int hiddenSize = 128;
    const int outputSize = 10;    // 10 digits
    const float learningRate = 0.01f;

    // Allocate device memory
    float *d_input, *d_hidden, *d_output;
    float *d_w1, *d_w2, *d_b1, *d_b2;
    float *d_delta_hidden, *d_delta_output;

    CUDA_CHECK(cudaMalloc(&d_input, inputSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hidden, hiddenSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, outputSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w1, inputSize * hiddenSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w2, hiddenSize * outputSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b1, hiddenSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b2, outputSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_delta_hidden, hiddenSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_delta_output, outputSize * sizeof(float)));

    // Initialize with random weights (simplified - would use cuRAND in production)
    float *h_w1 = (float*)malloc(inputSize * hiddenSize * sizeof(float));
    float *h_w2 = (float*)malloc(hiddenSize * outputSize * sizeof(float));
    for (int i = 0; i < inputSize * hiddenSize; i++) {
        h_w1[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.1f;
    }
    for (int i = 0; i < hiddenSize * outputSize; i++) {
        h_w2[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.1f;
    }

    CUDA_CHECK(cudaMemcpy(d_w1, h_w1, inputSize * hiddenSize * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w2, h_w2, hiddenSize * outputSize * sizeof(float),
                          cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threads = 256;

    // Forward pass
    cudaEventRecord(start);
    forwardLayer<<<(hiddenSize + threads - 1) / threads, threads>>>
        (d_input, d_w1, d_b1, d_hidden, inputSize, hiddenSize);
    forwardLayer<<<(outputSize + threads - 1) / threads, threads>>>
        (d_hidden, d_w2, d_b2, d_output, hiddenSize, outputSize);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float forwardTime;
    cudaEventElapsedTime(&forwardTime, start, stop);

    // Backward pass
    cudaEventRecord(start);
    backwardLayer<<<(hiddenSize + threads - 1) / threads, threads>>>
        (d_hidden, d_w2, d_delta_output, d_delta_hidden, hiddenSize, outputSize);
    updateWeights<<<(hiddenSize + threads - 1) / threads, threads>>>
        (d_w1, d_input, d_delta_hidden, inputSize, hiddenSize, learningRate);
    updateWeights<<<(outputSize + threads - 1) / threads, threads>>>
        (d_w2, d_hidden, d_delta_output, hiddenSize, outputSize, learningRate);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float backwardTime;
    cudaEventElapsedTime(&backwardTime, start, stop);

    printf("Network architecture: %d -> %d -> %d\n",
           inputSize, hiddenSize, outputSize);
    printf("Forward pass:  %.3f ms\n", forwardTime);
    printf("Backward pass: %.3f ms\n", backwardTime);
    printf("Total:         %.3f ms\n", forwardTime + backwardTime);
    printf("\nParameters: %d weights\n",
           inputSize * hiddenSize + hiddenSize * outputSize);

    free(h_w1);
    free(h_w2);
    cudaFree(d_input);
    cudaFree(d_hidden);
    cudaFree(d_output);
    cudaFree(d_w1);
    cudaFree(d_w2);
    cudaFree(d_b1);
    cudaFree(d_b2);
    cudaFree(d_delta_hidden);
    cudaFree(d_delta_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
