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

struct Body {
    float x, y, z;     // position
    float vx, vy, vz;  // velocity
    float mass;
};

__global__ void computeForces(Body *bodies, float *fx, float *fy, float *fz, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float force_x = 0.0f, force_y = 0.0f, force_z = 0.0f;
    const float G = 6.67430e-11f;  // Gravitational constant
    const float softening = 1e-9f;  // Softening factor to avoid singularities

    for (int j = 0; j < n; j++) {
        if (i != j) {
            float dx = bodies[j].x - bodies[i].x;
            float dy = bodies[j].y - bodies[i].y;
            float dz = bodies[j].z - bodies[i].z;

            float distSqr = dx*dx + dy*dy + dz*dz + softening;
            float dist = sqrtf(distSqr);
            float force = G * bodies[i].mass * bodies[j].mass / distSqr;

            force_x += force * dx / dist;
            force_y += force * dy / dist;
            force_z += force * dz / dist;
        }
    }

    fx[i] = force_x;
    fy[i] = force_y;
    fz[i] = force_z;
}

__global__ void updateBodies(Body *bodies, float *fx, float *fy, float *fz,
                              int n, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Update velocities (F = ma => a = F/m)
    float ax = fx[i] / bodies[i].mass;
    float ay = fy[i] / bodies[i].mass;
    float az = fz[i] / bodies[i].mass;

    bodies[i].vx += ax * dt;
    bodies[i].vy += ay * dt;
    bodies[i].vz += az * dt;

    // Update positions
    bodies[i].x += bodies[i].vx * dt;
    bodies[i].y += bodies[i].vy * dt;
    bodies[i].z += bodies[i].vz * dt;
}

int main() {
    printf("=== N-Body Gravitational Simulation ===\n\n");

    const int N = 4096;  // Number of bodies
    const float dt = 0.01f;  // Time step
    const int steps = 100;

    // Allocate host memory
    Body *h_bodies = (Body*)malloc(N * sizeof(Body));

    // Initialize with random positions and masses
    for (int i = 0; i < N; i++) {
        h_bodies[i].x = (rand() / (float)RAND_MAX - 0.5f) * 1e10f;
        h_bodies[i].y = (rand() / (float)RAND_MAX - 0.5f) * 1e10f;
        h_bodies[i].z = (rand() / (float)RAND_MAX - 0.5f) * 1e10f;
        h_bodies[i].vx = h_bodies[i].vy = h_bodies[i].vz = 0.0f;
        h_bodies[i].mass = 1e20f + rand() / (float)RAND_MAX * 1e20f;
    }

    // Allocate device memory
    Body *d_bodies;
    float *d_fx, *d_fy, *d_fz;
    CUDA_CHECK(cudaMalloc(&d_bodies, N * sizeof(Body)));
    CUDA_CHECK(cudaMalloc(&d_fx, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fy, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fz, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_bodies, h_bodies, N * sizeof(Body),
                          cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Simulation loop
    for (int step = 0; step < steps; step++) {
        computeForces<<<blocks, threads>>>(d_bodies, d_fx, d_fy, d_fz, N);
        updateBodies<<<blocks, threads>>>(d_bodies, d_fx, d_fy, d_fz, N, dt);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // Copy final positions back
    CUDA_CHECK(cudaMemcpy(h_bodies, d_bodies, N * sizeof(Body),
                          cudaMemcpyDeviceToHost));

    printf("Bodies: %d\n", N);
    printf("Time steps: %d\n", steps);
    printf("Total time: %.2f ms\n", ms);
    printf("Time per step: %.2f ms\n", ms / steps);
    printf("Interactions/sec: %.2f billion\n",
           (long long)N * N * steps / (ms / 1000.0) / 1e9);

    printf("\nSample final positions:\n");
    for (int i = 0; i < 5; i++) {
        printf("  Body %d: (%.2e, %.2e, %.2e)\n",
               i, h_bodies[i].x, h_bodies[i].y, h_bodies[i].z);
    }

    free(h_bodies);
    cudaFree(d_bodies);
    cudaFree(d_fx);
    cudaFree(d_fy);
    cudaFree(d_fz);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
