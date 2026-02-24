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

struct Atom {
    float x, y, z;     // position
    float vx, vy, vz;  // velocity
    float fx, fy, fz;  // force
};

__global__ void computeLennardJonesForces(Atom *atoms, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float epsilon = 1.0f;  // Energy parameter
    const float sigma = 1.0f;    // Distance parameter
    const float cutoff = 2.5f * sigma;

    float force_x = 0.0f, force_y = 0.0f, force_z = 0.0f;

    for (int j = 0; j < n; j++) {
        if (i != j) {
            float dx = atoms[j].x - atoms[i].x;
            float dy = atoms[j].y - atoms[i].y;
            float dz = atoms[j].z - atoms[i].z;

            float r2 = dx*dx + dy*dy + dz*dz;

            if (r2 < cutoff * cutoff) {
                float r2inv = 1.0f / r2;
                float r6inv = r2inv * r2inv * r2inv;
                float sigma6 = sigma * sigma * sigma * sigma * sigma * sigma;

                // Lennard-Jones potential: V(r) = 4ε[(σ/r)^12 - (σ/r)^6]
                // Force = -dV/dr
                float force = 24.0f * epsilon * r2inv * sigma6 * r6inv *
                             (2.0f * sigma6 * r6inv - 1.0f);

                force_x += force * dx;
                force_y += force * dy;
                force_z += force * dz;
            }
        }
    }

    atoms[i].fx = force_x;
    atoms[i].fy = force_y;
    atoms[i].fz = force_z;
}

__global__ void integrateVerlet(Atom *atoms, int n, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float mass = 1.0f;

    // Velocity Verlet integration
    // v(t+dt/2) = v(t) + f(t)/(2m) * dt
    atoms[i].vx += 0.5f * atoms[i].fx / mass * dt;
    atoms[i].vy += 0.5f * atoms[i].fy / mass * dt;
    atoms[i].vz += 0.5f * atoms[i].fz / mass * dt;

    // x(t+dt) = x(t) + v(t+dt/2) * dt
    atoms[i].x += atoms[i].vx * dt;
    atoms[i].y += atoms[i].vy * dt;
    atoms[i].z += atoms[i].vz * dt;
}

int main() {
    printf("=== Molecular Dynamics: Lennard-Jones ===\n\n");

    const int N = 2048;  // Number of atoms
    const float dt = 0.001f;
    const int steps = 100;

    Atom *h_atoms = (Atom*)malloc(N * sizeof(Atom));

    // Initialize atoms in a grid
    int cubeSize = (int)ceil(pow(N, 1.0/3.0));
    int idx = 0;
    for (int ix = 0; ix < cubeSize && idx < N; ix++) {
        for (int iy = 0; iy < cubeSize && idx < N; iy++) {
            for (int iz = 0; iz < cubeSize && idx < N; iz++, idx++) {
                h_atoms[idx].x = ix * 1.5f;
                h_atoms[idx].y = iy * 1.5f;
                h_atoms[idx].z = iz * 1.5f;
                h_atoms[idx].vx = (rand() / (float)RAND_MAX - 0.5f) * 0.1f;
                h_atoms[idx].vy = (rand() / (float)RAND_MAX - 0.5f) * 0.1f;
                h_atoms[idx].vz = (rand() / (float)RAND_MAX - 0.5f) * 0.1f;
                h_atoms[idx].fx = h_atoms[idx].fy = h_atoms[idx].fz = 0.0f;
            }
        }
    }

    Atom *d_atoms;
    CUDA_CHECK(cudaMalloc(&d_atoms, N * sizeof(Atom)));
    CUDA_CHECK(cudaMemcpy(d_atoms, h_atoms, N * sizeof(Atom),
                          cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int step = 0; step < steps; step++) {
        computeLennardJonesForces<<<blocks, threads>>>(d_atoms, N);
        integrateVerlet<<<blocks, threads>>>(d_atoms, N, dt);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    CUDA_CHECK(cudaMemcpy(h_atoms, d_atoms, N * sizeof(Atom),
                          cudaMemcpyDeviceToHost));

    printf("Atoms: %d\n", N);
    printf("Time steps: %d\n", steps);
    printf("Total time: %.2f ms\n", ms);
    printf("Time per step: %.3f ms\n", ms / steps);
    printf("Interactions/step: %d\n", N * N);

    printf("\nSample final positions:\n");
    for (int i = 0; i < 5; i++) {
        printf("  Atom %d: (%.2f, %.2f, %.2f)\n",
               i, h_atoms[i].x, h_atoms[i].y, h_atoms[i].z);
    }

    free(h_atoms);
    cudaFree(d_atoms);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
