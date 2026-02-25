// N-Body Simulation
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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


__global__ void nbodyKernel(float *pos, float *vel, int n, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float px = pos[i*3], py = pos[i*3+1], pz = pos[i*3+2];
    float fx = 0.0f, fy = 0.0f, fz = 0.0f;

    for (int j = 0; j < n; j++) {
        float dx = pos[j*3] - px;
        float dy = pos[j*3+1] - py;
        float dz = pos[j*3+2] - pz;
        float dist = sqrt(dx*dx + dy*dy + dz*dz + 1e-10f);
        float f = 1.0f / (dist * dist * dist);
        fx += dx * f;
        fy += dy * f;
        fz += dz * f;
    }

    vel[i*3] += fx * dt;
    vel[i*3+1] += fy * dt;
    vel[i*3+2] += fz * dt;
    pos[i*3] += vel[i*3] * dt;
    pos[i*3+1] += vel[i*3+1] * dt;
    pos[i*3+2] += vel[i*3+2] * dt;
}

int main() {
    printf("=== N-Body Simulation ===\n\n");

    int n = 4096;
    float dt = 0.01f;
    int steps = 10;

    float *h_pos = (float*)malloc(n * 3 * sizeof(float));
    float *h_vel = (float*)malloc(n * 3 * sizeof(float));

    for (int i = 0; i < n * 3; i++) {
        h_pos[i] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        h_vel[i] = 0.0f;
    }

    float *d_pos, *d_vel;
    CUDA_CHECK(cudaMalloc(&d_pos, n * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vel, n * 3 * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_pos, h_pos, n * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel, h_vel, n * 3 * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int step = 0; step < steps; step++) {
        nbodyKernel<<<(n+255)/256, 256>>>(d_pos, d_vel, n, dt);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    printf("Bodies: %d\n", n);
    printf("Steps: %d\n", steps);
    printf("Total time: %.2f ms\n", ms);
    printf("Time per step: %.2f ms\n", ms / steps);
    printf("Interactions/sec: %.2f million\n", (n * n * steps / 1e6) / (ms / 1000.0));

    free(h_pos); free(h_vel);
    cudaFree(d_pos); cudaFree(d_vel);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
